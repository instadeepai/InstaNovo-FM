#!/usr/bin/env python
"""theoretical_spectra.py
========================

Utility functions for generating and matching theoretical fragment ion spectra
for peptides in mass spectrometry data analysis.

Core functionality:
1. Generate theoretical fragment ion spectra (b, y, a, c, x, z ions)
2. Match experimental peaks to theoretical fragments
3. Detect custom ion types (diagnostic, reporter, immonium, etc.)
4. Bulk annotation of spectra DataFrames

Designed for AI model training pipelines and multi-instrument/multi-fragmentation
proteomics workflows.

Dependencies:
    - pyopenms (>=3.4.0) - for pyopenms engine
    - rustyms (>=0.10.0) - for annotator/rustyms engine (optional)
    - numpy
    - polars
    - tqdm (optional, for progress bars)

Example Usage:
    ```python
    from instanovo_fm.utils.theoretical_spectra import (
        generate_theoretical_spectrum,
        annotate_dataframe,
        detect_custom_ions
    )
    
    # Generate theoretical spectrum
    mz, annotations = generate_theoretical_spectrum(
        peptide="PEPTIDE",
        precursor_charge=2,
        ion_types=["b", "y"],
        add_isotopes=True,
        isotope_model="fine"
    )
    
    # Annotate experimental data
    annotated_df = annotate_dataframe(
        df,
        ppm_tol=10.0,
        ion_types=["b", "y"],
        add_isotopes=True,
        detect_custom=True,
        custom_ions={"immonium": [110.0713, 120.0813]}
    )
    ```
"""
from __future__ import annotations

import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import polars as pl

try:
    from pyopenms import AASequence, MSSpectrum, Param, TheoreticalSpectrumGenerator
except ImportError as e:
    raise ImportError(
        "PyOpenMS is required. Install with: pip install pyopenms>=3.4.0"
    ) from e

# Lazy import for rustyms (optional dependency)
RUSTYMS_AVAILABLE = False
try:
    import rustyms
    RUSTYMS_AVAILABLE = True
except ImportError:
    rustyms = None

# Physical constants
PROTON_MASS = 1.007276466812  # monoisotopic proton mass (Da)
DEFAULT_CID_DA_TOL: float = 0.8  # Da tolerance for low-resolution CID (ion-trap MS2)

# Default custom ion library
# All values are observed m/z in positive-mode MS2 (i.e. [M+H]+ or reporter
# cation m/z). Neutral losses belong in LOSS_MASSES, not here.
DEFAULT_CUSTOM_IONS: Dict[str, List[float]] = {
    # --- Glycan oxonium ions (glycopeptides) ---
    "glycan_HexNAc": [204.08665],
    "glycan_HexNAc_fragment_126": [126.05495],
    "glycan_HexNAc_fragment_138": [138.05495],
    "glycan_HexNAc_fragment_144": [144.06552],
    "glycan_HexNAc_fragment_168": [168.06552],
    "glycan_HexNAc_fragment_186": [186.07608],
    "glycan_HexHexNAc": [366.13947],
    "glycan_NeuAc": [274.09213, 292.10269],
    "glycan_dHex": [147.0652],
    "glycan_Hex": [163.06010],
    "glycan_HexNAc2": [407.16600],
    "glycan_Hex2HexNAc": [528.19228],
    "glycan_HexHexNAcNeuAc": [657.23480],
    # --- Phosphorylation reporter ---
    "immonium_PhosphoTyr": [216.0426],
    # --- Common immonium ions (amino-acid fingerprints) ---
    "immonium_Ser": [60.0444],
    "immonium_Thr": [74.0600],
    "immonium_Pro": [70.0651],
    "immonium_Val": [72.0808],
    "immonium_Ile_Leu": [86.0964],
    "immonium_Asn": [87.0553],
    "immonium_Asp": [88.0393],
    "immonium_Gln": [101.0709],
    "immonium_Glu": [102.0550],
    "immonium_Lys": [101.1073],
    "immonium_Met": [104.0528],
    "immonium_His": [110.0713],
    "immonium_Phe": [120.0813],
    "immonium_Arg": [129.1135],
    "immonium_Tyr": [136.0757],
    "immonium_Trp": [159.0922],
    # --- PTM-specific immoniums ---
    "immonium_AcetylLys": [126.0913],
    "immonium_MonoMethylLys": [98.0964],
    "immonium_DiMethylLys": [112.1121],
    # --- Immonium-related diagnostic ions (Falick et al. 1993) ---
    # Secondary fragments from amino acid side chains, commonly observed
    # in the low-mass region of HCD spectra alongside primary immonium ions.
    "related_V_41": [41.039],          # C3H5+ allyl cation
    "related_V_55": [55.055],          # C4H7+
    "related_V_69": [69.070],          # C5H9+
    "related_IL_44": [44.050],         # CO-NH2=CH2+ (Leu/Ile shared)
    "related_IL_72": [72.081],         # immonium - CO (Leu/Ile shared)
    "related_N_70": [70.029],          # Asn immonium - NH3
    "related_D_70": [70.029],          # Asp immonium - H2O
    "related_Q_56": [56.050],          # Gln-related
    "related_Q_84": [84.045],          # Gln immonium - NH3
    "related_Q_129": [129.066],        # Gln immonium + CO
    "related_K_70": [70.066],          # Lys-related
    "related_K_84": [84.081],          # Lys immonium - NH3
    "related_K_112": [112.076],        # Lys-related
    "related_K_129": [129.102],        # Lys immonium + CO
    "related_M_61": [61.011],          # Met immonium - CH2S
    "related_H_82": [82.053],          # His immonium - CO
    "related_H_121": [121.076],        # His-related
    "related_H_123": [123.055],        # His-related
    "related_H_138": [138.066],        # His immonium + CO
    "related_H_166": [166.061],        # His-related
    "related_F_91": [91.054],          # Tropylium C7H7+ (Phe)
    "related_Y_91": [91.054],          # Tropylium C7H7+ (Tyr, same mass as Phe)
    "related_Y_107": [107.049],        # Hydroxytropylium C7H7O+
    "related_R_59": [59.048],          # Arg-related
    "related_R_70": [70.066],          # Arg-related
    "related_R_73": [73.064],          # Arg-related
    "related_R_87": [87.056],          # Arg-related
    "related_R_100": [100.087],        # Arg-related
    "related_R_112": [112.087],        # Arg-related
    "related_W_77": [77.039],          # Phenyl cation C6H5+
    "related_W_117": [117.058],        # Indolium C8H7N+
    "related_W_130": [130.066],        # 3-methylindolium C9H8N+
    "related_W_132": [132.081],        # Trp-related
    "related_W_170": [170.060],        # Trp-related
    "related_W_171": [171.092],        # Trp-related
    # --- TMT 11-plex reporter ions ---
    "TMT_126": [126.1277],
    "TMT_127N": [127.1248],
    "TMT_127C": [127.1311],
    "TMT_128N": [128.1281],
    "TMT_128C": [128.1344],
    "TMT_129N": [129.1315],
    "TMT_129C": [129.1378],
    "TMT_130N": [130.1348],
    "TMT_130C": [130.1411],
    "TMT_131N": [131.1382],
    "TMT_131C": [131.1415],
    # --- TMTpro 16/18-plex reporter ions ---
    "TMTpro_132N": [132.1385],
    "TMTpro_132C": [132.1449],
    "TMTpro_133N": [133.1419],
    "TMTpro_133C": [133.1452],
    "TMTpro_134N": [134.1452],
    "TMTpro_134C": [134.1486],
    "TMTpro_135N": [135.1486],
    # --- iTRAQ 8-plex reporter ions ---
    "iTRAQ_113": [113.1078],
    "iTRAQ_114": [114.1112],
    "iTRAQ_115": [115.1083],
    "iTRAQ_116": [116.1116],
    "iTRAQ_117": [117.1150],
    "iTRAQ_118": [118.1121],
    "iTRAQ_119": [119.1154],
    "iTRAQ_121": [121.1220],
}

__all__ = [
    "generate_theoretical_spectrum",
    "match_theoretical_to_experimental",
    "detect_custom_ions",
    "annotate_dataframe",
    "match_with_conditional_features",
    "compute_theoretical_precursor_mz",
    "DEFAULT_CUSTOM_IONS",
    "PROTON_MASS",
    "DEFAULT_CID_DA_TOL",
    "_da_tol_for_fragmentation",
]

# =============================================================================
# Helper Functions
# =============================================================================


def compute_theoretical_precursor_mz(peptide: str, charge: int) -> float:
    """Compute theoretical precursor m/z from peptide string and charge.

    Parameters
    ----------
    peptide : str
        Peptide sequence in PyOpenMS-compatible format.
    charge : int
        Precursor charge state.

    Returns
    -------
    float
        Theoretical precursor m/z value.
    """
    aa_seq = AASequence.fromString(peptide)
    neutral_mass = aa_seq.getMonoWeight()
    return (neutral_mass + charge * PROTON_MASS) / charge


def _ppm_window(mz: float, ppm: float) -> Tuple[float, float]:
    """Calculate m/z tolerance window for given ppm tolerance.

    Parameters
    ----------
    mz : float
        Target m/z value
    ppm : float
        Tolerance in parts per million

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    delta = mz * ppm * 1e-6
    return mz - delta, mz + delta


def _da_window(mz: float, da: float) -> Tuple[float, float]:
    """Calculate m/z tolerance window for given absolute Da tolerance.

    Parameters
    ----------
    mz : float
        Target m/z value
    da : float
        Tolerance in Daltons

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    return mz - da, mz + da


def _da_tol_for_fragmentation(
    frag_type: Optional[str],
    cid_da_tol: float = DEFAULT_CID_DA_TOL,
) -> Optional[float]:
    """Return Da tolerance for low-res CID; None for high-res instruments.

    CID on ion-trap analyzers has fragment errors of ~0.3–0.5 Da, so ppm
    matching is inappropriate.  HCD, HCID, ETD are high-resolution and should
    keep ppm matching (return None).

    Args:
        frag_type: Fragmentation type string from data (e.g. "CID", "HCD").
        cid_da_tol: Da tolerance to use for CID.  Callers can pass None to
                    disable auto-selection entirely.
    Returns:
        cid_da_tol if frag_type is CID, else None.
    """
    if not frag_type:
        return None
    if str(frag_type).strip().upper() == "CID":
        return cid_da_tol
    return None


def _validate_peptide_sequence(peptide: str) -> str:
    """Validate and clean peptide sequence.
    
    Parameters
    ----------
    peptide : str
        Peptide sequence
        
    Returns
    -------
    str
        Cleaned peptide sequence
        
    Raises
    ------
    ValueError
        If sequence is invalid
    """
    peptide = peptide.strip()
    if not peptide:
        raise ValueError("Peptide sequence must be non-empty.")
    
    # Try parsing with PyOpenMS to validate
    try:
        AASequence.fromString(peptide)
    except Exception as e:
        raise ValueError(f"Invalid peptide sequence '{peptide}': {e}") from e
    
    return peptide


def _ensure_sorted(
    mz: np.ndarray, intensity: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Ensure m/z array is sorted, optionally with intensities.
    
    Parameters
    ----------
    mz : np.ndarray
        m/z values
    intensity : Optional[np.ndarray], optional
        Intensity values, by default None
        
    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Sorted m/z array, or (mz, intensity) tuple if intensity provided
    """
    if np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz = mz[order]
        if intensity is not None:
            intensity = intensity[order]
            return mz, intensity
    
    if intensity is not None:
        return mz, intensity
    return mz


# =============================================================================
# 0. Engine Selection and Fragmentation Type Mapping
# =============================================================================


def _check_rustyms_available(engine: str) -> None:
    """Check if rustyms is available when annotator engine is selected.
    
    Args:
        engine: Engine name ("pyopenms" or "annotator")
        
    Raises:
        ImportError: If rustyms engine is selected but rustyms is not installed
    """
    if engine in ("annotator", "rustyms") and not RUSTYMS_AVAILABLE:
        raise ImportError(
            "rustyms engine selected but rustyms is not installed. "
            "Please install rustyms (pip install rustyms>=0.10.0) or switch engine to pyopenms."
        )


def _map_fragmentation_type_to_rustyms(frag_type: Optional[str], acquisition: Optional[str] = None) -> str:
    """Map fragmentation type and acquisition mode to rustyms fragmentation model.
    
    Args:
        frag_type: Fragmentation type (e.g., "HCD", "CID", "ETD")
        acquisition: Acquisition mode (e.g., "DIA", "DDA")
        
    Returns:
        rustyms fragmentation model name
    """
    if not frag_type:
        return "hcd"  # Default to HCD
    
    frag_type_upper = str(frag_type).strip().upper()
    
    # Map common fragmentation types
    if frag_type_upper in ("HCD", "HCID"):
        return "hcd"
    elif frag_type_upper == "CID":
        return "cid"
    elif frag_type_upper in ("ETD", "ECD"):
        return "etd"
    elif frag_type_upper == "UVPD":
        return "uvpd"
    else:
        # Default to HCD for unknown types
        return "hcd"


def _ion_types_to_rustyms(ion_types: Sequence[str]) -> List[str]:
    """Convert ion type sequence to rustyms-compatible format.
    
    Args:
        ion_types: Sequence of ion types (e.g., ["b", "y"])
        
    Returns:
        List of rustyms-compatible ion type strings
    """
    rustyms_ion_types = []
    for ion in ion_types:
        ion_lower = str(ion).lower().strip()
        if ion_lower in ("a", "b", "c", "x", "y", "z"):
            rustyms_ion_types.append(ion_lower)
    return rustyms_ion_types if rustyms_ion_types else ["b", "y"]  # Default


def _extract_rustyms_fragment_info(fragment) -> Dict[str, Any]:
    """Extract detailed information from a rustyms Fragment object.
    
    Args:
        fragment: rustyms Fragment object
        
    Returns:
        Dictionary with fragment information:
        - mz: Calculated m/z value
        - mass: Monoisotopic mass
        - charge: Charge state
        - ion_type: Ion type string (e.g., "b3")
        - neutral_loss: Neutral loss string (e.g., "H2O1") or empty
        - annotation: Formatted annotation string
        - formula: Molecular formula string (if available)
    """
    PROTON_MASS_RUSTYMS = 1.007276466812
    
    # Calculate m/z
    mass = fragment.formula.monoisotopic_mass() if fragment.formula else 0.0
    charge = fragment.charge
    mz = (mass + charge * PROTON_MASS_RUSTYMS) / charge if charge > 0 else 0.0
    
    # Extract ion type from repr
    frag_repr = repr(fragment)
    ion_str = ""
    if "ion='" in frag_repr:
        start = frag_repr.find("ion='") + 5
        end = frag_repr.find("'", start)
        if end > start:
            ion_str = frag_repr[start:end]
    
    # Extract neutral loss
    neutral_loss_str = ""
    if hasattr(fragment, 'neutral_loss'):
        try:
            nl_repr = repr(fragment.neutral_loss)
            if '["-' in nl_repr:
                loss_start = nl_repr.find('["-') + 3
                loss_end = nl_repr.find('"]', loss_start)
                if loss_end > loss_start:
                    neutral_loss_str = nl_repr[loss_start:loss_end]
        except:
            pass
    
    # Fallback to repr parsing
    if not neutral_loss_str and "neutral_loss=" in frag_repr and '["-' in frag_repr:
        loss_start = frag_repr.find('["-') + 3
        loss_end = frag_repr.find('"]', loss_start)
        if loss_end > loss_start:
            neutral_loss_str = frag_repr[loss_start:loss_end]
    
    # Format annotation
    charge_str = "+" * charge
    if neutral_loss_str:
        annotation = f"{ion_str}-{neutral_loss_str}{charge_str}"
    else:
        annotation = f"{ion_str}{charge_str}" if ion_str else f"frag{charge_str}"
    
    # Get formula string
    formula_str = ""
    if fragment.formula:
        try:
            formula_str = str(fragment.formula)
        except:
            pass
    
    return {
        "mz": float(mz),
        "mass": float(mass),
        "charge": int(charge),
        "ion_type": ion_str,
        "neutral_loss": neutral_loss_str,
        "annotation": annotation,
        "formula": formula_str,
    }


# =============================================================================
# 1. Theoretical Spectrum Generation
# =============================================================================


def generate_theoretical_spectrum(
    peptide: str,
    precursor_charge: int = 2,
    ion_types: Sequence[str] = ("b", "y"),
    max_charge: Optional[int] = None,
    add_losses: bool = False,
    loss_types: Sequence[str] = ("H2O", "NH3"),
    add_isotopes: bool = False,
    max_isotope: int = 3,
    isotope_model: Literal["coarse", "fine"] = "coarse",
    max_isotope_probability: float = 0.95,
    add_precursor: bool = False,
    precursor_isotopes: bool = False,
    max_precursor_isotope: int = 3,
    add_custom_ions: bool = False,
    custom_ions: Optional[Dict[str, List[float]]] = None,
    sort_by_mz: bool = True,
    engine: str = "pyopenms",
    fragmentation_type: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Generate theoretical fragment ion spectrum for a peptide.

    Supports two engines:
    - "pyopenms": Uses PyOpenMS TheoreticalSpectrumGenerator (default)
    - "annotator" or "rustyms": Uses rustyms/annotator library

    Computes fragment m/z values for specified ion types, with optional neutral
    losses, isotopes, and custom ion types.

    Parameters
    ----------
    peptide : str
        Peptide sequence (supports modifications, e.g., "PEPTIDEM(Oxidation)")
    precursor_charge : int, default=2
        Precursor charge state
    ion_types : Sequence[str], default=("b", "y")
        Fragment ion types to generate. Options: "a", "b", "c", "x", "y", "z"
    max_charge : Optional[int], default=None
        Maximum fragment charge state (defaults to precursor_charge)
    add_losses : bool, default=False
        Include neutral losses (H₂O, NH₃)
    add_isotopes : bool, default=False
        Include isotopic peaks
    max_isotope : int, default=3
        For coarse model: maximum isotope number (+1, +2, +3, etc.)
        Ignored for fine model
    isotope_model : {"coarse", "fine"}, default="coarse"
        Isotope generation model:
        - "coarse": Simple +1.003 Da spacing up to max_isotope
        - "fine": Accurate elemental isotope distribution
    max_isotope_probability : float, default=0.95
        For fine model: cumulative probability threshold (0.0-1.0)
        Ignored for coarse model
    add_precursor : bool, default=False
        Include precursor m/z (useful for all-ion fragmentation)
    precursor_isotopes : bool, default=False
        Append isotopic variants (M+1 … M+max_precursor_isotope) for each
        precursor entry.  Only effective when *add_precursor* is True.
    max_precursor_isotope : int, default=3
        Maximum isotope offset for precursor ions.
    add_custom_ions : bool, default=False
        Append custom ion m/z values to theoretical spectrum
    custom_ions : Optional[Dict[str, List[float]]], default=None
        Dictionary mapping ion names to m/z values.
        If None and add_custom_ions=True, uses DEFAULT_CUSTOM_IONS.
    sort_by_mz : bool, default=True
        Sort output by m/z value
    engine : str, default="pyopenms"
        Engine to use: "pyopenms" or "annotator"/"rustyms"
    fragmentation_type : Optional[str], default=None
        Fragmentation type (e.g., "HCD", "CID", "ETD") - used by rustyms engine
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (mz_array, annotations)
        - mz_array: 1D array of theoretical m/z values
        - annotations: List of ion annotations (e.g., "b3+", "y5++")
        
    Raises
    ------
    ValueError
        If peptide sequence is invalid or ion_types contain unknown values
    ImportError
        If rustyms engine is selected but rustyms is not installed
        
    Examples
    --------
    >>> mz, ann = generate_theoretical_spectrum("PEPTIDE", precursor_charge=2)
    >>> mz[:3]
    array([97.0284, 226.0710, 325.1394])
    >>> ann[:3]
    ['b1+', 'b2+', 'b3+']
    
    >>> # With modifications
    >>> mz, ann = generate_theoretical_spectrum(
    ...     "PEPTIDEM(Oxidation)K",
    ...     ion_types=["b", "y"],
    ...     add_losses=True,
    ...     add_isotopes=True,
    ...     isotope_model="fine"
    ... )
    """
    # Dispatch to appropriate engine
    if engine in ("annotator", "rustyms"):
        # Use rustyms engine
        return generate_theoretical_spectrum_rustyms(
            peptide=peptide,
            precursor_charge=precursor_charge,
            ion_types=ion_types,
            max_charge=max_charge,
            fragmentation_type=fragmentation_type,
            add_losses=add_losses,
            add_isotopes=add_isotopes,
            sort_by_mz=sort_by_mz,
        )
    elif engine == "pyopenms":
        # Use PyOpenMS engine (existing implementation)
        pass  # Continue with existing code
    else:
        raise ValueError(f"Unknown engine: {engine}. Must be 'pyopenms' or 'annotator'")
    
    peptide = _validate_peptide_sequence(peptide)
    
    if max_charge is None:
        max_charge = precursor_charge
    
    # Parse peptide with PyOpenMS
    aa_seq = AASequence.fromString(peptide)
    
    # Configure theoretical spectrum generator
    tsg = TheoreticalSpectrumGenerator()
    params = Param()
    params.setValue("add_metainfo", "true")
    
    # Disable all ion types, then enable requested ones
    ion_type_keys = {
        "a": "add_a_ions",
        "b": "add_b_ions",
        "c": "add_c_ions",
        "x": "add_x_ions",
        "y": "add_y_ions",
        "z": "add_z_ions",
    }
    
    for key in ion_type_keys.values():
        params.setValue(key, "false")
    params.setValue("add_losses", "false")
    
    # Enable requested ion types
    for ion in ion_types:
        ion_lower = ion.lower()
        if ion_lower not in ion_type_keys:
            raise ValueError(
                f"Unknown ion type '{ion}'. Valid options: {list(ion_type_keys.keys())}"
            )
        params.setValue(ion_type_keys[ion_lower], "true")
    
    # Neutral losses
    if add_losses:
        params.setValue("add_losses", "true")
    
    # Isotope configuration (OpenMS 3.x)
    if add_isotopes:
        if isotope_model not in ("coarse", "fine"):
            raise ValueError(
                f"isotope_model must be 'coarse' or 'fine', got '{isotope_model}'"
            )
        
        params.setValue("isotope_model", isotope_model)
        if isotope_model == "coarse":
            if max_isotope < 1:
                raise ValueError(f"max_isotope must be >= 1 for coarse model, got {max_isotope}")
            params.setValue("max_isotope", int(max_isotope))
        elif isotope_model == "fine":
            if not (0 < max_isotope_probability <= 1):
                raise ValueError(
                    f"max_isotope_probability must be in (0, 1] for fine model, got {max_isotope_probability}"
                )
            params.setValue("max_isotope_probability", float(max_isotope_probability))
    else:
        params.setValue("isotope_model", "none")
    
    tsg.setParameters(params)
    
    # Generate spectrum
    spec = MSSpectrum()
    tsg.getSpectrum(spec, aa_seq, 1, max_charge)
    
    # Extract m/z and annotations
    mz = np.asarray(spec.get_peaks()[0], dtype=np.float64)
    
    # Look for "IonName" or "IonNames" string data array (PyOpenMS uses "IonNames")
    annotations = []
    ion_name_array = None
    
    for string_array in spec.getStringDataArrays():
        if string_array.getName() in ("IonName", "IonNames"):
            ion_name_array = string_array
            break
    
    # Get charge array if available (needed for isotope detection)
    charge_array = None
    for int_array in spec.getIntegerDataArrays():
        if int_array.getName() == "Charges":
            charge_array = int_array
            break
    
    if ion_name_array is not None:
        # PyOpenMS returns bytes, need to decode properly
        raw_annotations = ion_name_array
        for i, s in enumerate(raw_annotations):
            if isinstance(s, bytes):
                # Decode bytes to string
                annotation = s.decode('utf-8')
            elif isinstance(s, str):
                # If already string but looks like "b'y1+'" (repr of bytes), clean it
                if s.startswith("b'") and s.endswith("'"):
                    annotation = s[2:-1]  # Remove b'...' wrapper
                else:
                    annotation = s
            else:
                annotation = str(s)
            
            annotations.append(annotation)
    else:
        # Fallback: use first string array if available, otherwise generate names
        if spec.getStringDataArrays():
            raw_annotations = spec.getStringDataArrays()[0]
            for s in raw_annotations:
                if isinstance(s, bytes):
                    annotations.append(s.decode('utf-8'))
                elif isinstance(s, str):
                    if s.startswith("b'") and s.endswith("'"):
                        annotations.append(s[2:-1])
                    else:
                        annotations.append(s)
                else:
                    annotations.append(str(s))
        else:
            annotations = [f"frag_{i}" for i in range(len(mz))]
    
    # Detect and annotate isotopes if isotopes were enabled
    # PyOpenMS doesn't distinguish isotopes in IonName, so we detect by m/z spacing
    if add_isotopes and len(mz) > 1:
        # Isotope spacing: ~1.003355 Da per charge (13C-12C mass difference)
        ISOTOPE_MASS_DIFF = 1.003355
        TOLERANCE = 0.01  # Tolerance for isotope detection
        
        # Create enhanced annotations with isotope notation
        enhanced_annotations = []
        # Track isotope numbers for each base annotation
        isotope_counts = {}  # Maps base_annotation -> current isotope number
        
        for i in range(len(mz)):
            base_annotation = annotations[i]
            
            # Determine charge for this peak
            charge = 1
            if charge_array is not None and i < charge_array.size():
                charge = charge_array[i]
            
            # Check if this is an isotope peak (m/z difference ~1.003/charge from previous peak)
            isotope_num = 0
            if i > 0:
                mz_diff = mz[i] - mz[i-1]
                expected_isotope_spacing = ISOTOPE_MASS_DIFF / charge
                
                # Check if m/z difference matches isotope spacing
                # and if base annotation matches previous peak (same ion type)
                if abs(mz_diff - expected_isotope_spacing) < TOLERANCE:
                    # Check if previous peak has the same base annotation
                    prev_base = annotations[i-1]
                    # Remove any existing isotope notation from previous
                    if '[' in prev_base:
                        prev_base = prev_base.split('[')[0]
                    
                    if base_annotation == prev_base:
                        # This is an isotope of the previous peak
                        # Increment isotope number for this base annotation
                        isotope_counts[base_annotation] = isotope_counts.get(base_annotation, 0) + 1
                        isotope_num = isotope_counts[base_annotation]
                    elif base_annotation in isotope_counts:
                        # Same base annotation but not consecutive - reset counter
                        isotope_counts[base_annotation] = 1
                        isotope_num = 1
                    else:
                        # New base annotation, start isotope counting
                        isotope_counts[base_annotation] = 1
                        isotope_num = 1
                else:
                    # Not an isotope - reset counter for this base annotation
                    if base_annotation in isotope_counts:
                        del isotope_counts[base_annotation]
            
            # Add isotope notation if detected
            if isotope_num > 0 and isotope_num <= max_isotope:
                enhanced_annotations.append(f"{base_annotation}[+{isotope_num}]")
            else:
                enhanced_annotations.append(base_annotation)
                # Reset counter if this is not an isotope
                if base_annotation in isotope_counts:
                    del isotope_counts[base_annotation]
        
        annotations = enhanced_annotations
    
    # Add precursor ion(s) if requested — PSI mzPAF format
    if add_precursor:
        prec_mz_list, prec_ann_list = _generate_precursor_ions(
            aa_seq=aa_seq,
            precursor_charge=precursor_charge,
            add_losses=add_losses,
            loss_types=loss_types,
            add_isotopes=precursor_isotopes,
            max_isotope=max_precursor_isotope,
        )
        mz = np.concatenate([mz, np.asarray(prec_mz_list, dtype=np.float64)])
        annotations = annotations + prec_ann_list
    
    # Add custom ions if requested
    if add_custom_ions:
        if custom_ions is None:
            custom_ions = DEFAULT_CUSTOM_IONS
        
        custom_mz = []
        custom_ann = []
        for ion_name, mz_values in custom_ions.items():
            for mz_val in mz_values:
                custom_mz.append(float(mz_val))
                custom_ann.append(f"custom:{ion_name}@{float(mz_val):.4f}")
        
        if custom_mz:
            mz = np.concatenate([mz, np.asarray(custom_mz, dtype=np.float64)])
            annotations = annotations + custom_ann
    
    # Sort by m/z if requested
    if sort_by_mz:
        order = np.argsort(mz)
        mz = mz[order]
        annotations = [annotations[i] for i in order]
    
    return mz, annotations


def generate_theoretical_spectrum_rustyms(
    peptide: str,
    precursor_charge: int = 2,
    ion_types: Sequence[str] = ("b", "y"),
    max_charge: Optional[int] = None,
    fragmentation_type: Optional[str] = None,
    add_losses: bool = False,
    add_isotopes: bool = False,
    sort_by_mz: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Generate theoretical fragment ion spectrum using rustyms/annotator.
    
    This is the rustyms-based implementation of theoretical spectrum generation.
    Uses rustyms to generate theoretical fragments.
    
    Parameters
    ----------
    peptide : str
        Peptide sequence (supports modifications in ProForma format)
    precursor_charge : int, default=2
        Precursor charge state
    ion_types : Sequence[str], default=("b", "y")
        Fragment ion types to generate (note: rustyms generates all types, we filter by ion_types)
    max_charge : Optional[int], default=None
        Maximum fragment charge state
    fragmentation_type : Optional[str], default=None
        Fragmentation type (e.g., "HCD", "CID", "ETD")
    add_losses : bool, default=False
        Include neutral losses (H₂O, NH₃) - handled by rustyms fragmentation model
    add_isotopes : bool, default=False
        Include isotopic peaks - handled by rustyms automatically
    sort_by_mz : bool, default=True
        Sort output by m/z value
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (mz_array, annotations)
    """
    _check_rustyms_available("annotator")
    
    if max_charge is None:
        max_charge = precursor_charge
    
    try:
        # Parse peptide with rustyms (supports ProForma format)
        peptidoform = rustyms.Peptidoform(peptide)
        
        # Map fragmentation type to rustyms model
        frag_model_name = _map_fragmentation_type_to_rustyms(fragmentation_type)
        frag_model_map = {
            "hcd": rustyms.FragmentationModel.CidHcd,
            "cid": rustyms.FragmentationModel.CidHcd,
            "etd": rustyms.FragmentationModel.Etd,
            "uvpd": rustyms.FragmentationModel.Uvpd,
        }
        frag_model = frag_model_map.get(frag_model_name, rustyms.FragmentationModel.CidHcd)
        
        # Generate theoretical fragments
        fragments = peptidoform.generate_theoretical_fragments(max_charge=max_charge, model=frag_model)
        
        if fragments is None or len(fragments) == 0:
            return np.array([], dtype=np.float64), []
        
        # Extract m/z and annotations from fragments
        mz_values = []
        annotations = []
        
        # Map ion types to filter (rustyms generates all types, we filter)
        requested_ion_types = set(ion_types)
        
        # PROTON_MASS constant for m/z calculation
        PROTON_MASS_RUSTYMS = 1.007276466812
        
        for fragment in fragments:
            # Extract fragment information using helper function
            frag_info = _extract_rustyms_fragment_info(fragment)
            
            # Skip if invalid
            if frag_info["mz"] == 0.0 or frag_info["charge"] == 0:
                continue
            
            # Filter by requested ion types if specified
            ion_type_char = frag_info["ion_type"][0].lower() if frag_info["ion_type"] else ""
            if requested_ion_types and ion_type_char not in requested_ion_types:
                continue
            
            mz_values.append(frag_info["mz"])
            annotations.append(frag_info["annotation"])
        
        mz_array = np.array(mz_values, dtype=np.float64)
        
        # Sort by m/z if requested
        if sort_by_mz and len(mz_array) > 0:
            order = np.argsort(mz_array)
            mz_array = mz_array[order]
            annotations = [annotations[i] for i in order]
        
        return mz_array, annotations
        
    except Exception as e:
        raise ValueError(f"rustyms failed to generate theoretical spectrum for '{peptide}': {e}") from e


# =============================================================================
# 2. Experimental-Theoretical Matching
# =============================================================================


def match_theoretical_to_experimental(
    exp_mz: np.ndarray,
    exp_intensity: np.ndarray,
    theo_mz: np.ndarray,
    ppm_tol: float = 10.0,
    da_tol: Optional[float] = None,
    use_closest: bool = False,
    theo_annotations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Match experimental peaks to theoretical fragment ions.

    Supports two tolerance modes:

    - **ppm** (default): window scales with m/z, appropriate for high-resolution
      instruments (Orbitrap, Q-TOF).  ``ppm_tol=10`` is a reasonable starting
      point; 5–20 ppm covers most Orbitrap HCD/CID data.
    - **Da** (``da_tol`` set): fixed absolute window, appropriate for
      low-resolution CID (ion-trap MS2) where a ppm window is far too narrow.
      Typical values: 0.3–0.5 Da for low-res CID, 0.02–0.05 Da when an
      absolute guarantee is preferred on high-res data.

    When ``da_tol`` is provided it takes precedence over ``ppm_tol``.

    Parameters
    ----------
    exp_mz : np.ndarray
        Experimental m/z values (will be sorted if not already)
    exp_intensity : np.ndarray
        Experimental intensity values
    theo_mz : np.ndarray
        Theoretical m/z values to match against
    ppm_tol : float, default=10.0
        Tolerance in parts per million.  Ignored when ``da_tol`` is set.
    da_tol : Optional[float], default=None
        Absolute tolerance in Daltons.  When provided, overrides ``ppm_tol``
        and uses a fixed-width window regardless of m/z.  Recommended for
        low-resolution CID data (start with 0.5 Da).
    use_closest : bool, default=False
        If True, match to closest peak by m/z instead of most intense
    theo_annotations : Optional[List[str]], default=None
        Fragment ion annotations (e.g., ["b1+", "y2+", "b3-H2O"])
        Must have same length as theo_mz
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "mask": boolean array marking matched experimental peaks
        - "match_idx": index of matched theoretical ion (-1 if unmatched)
        - "ppm_error": ppm error for each experimental peak (NaN if unmatched)
        - "matched_annotation": fragment annotation for each experimental peak (None if unmatched)
        - "matched_theo_mz": theoretical m/z for each experimental peak (NaN if unmatched)
        - "metrics": dict with summary statistics:
            - "n_matched": number of matched peaks
            - "frac_intensity": fraction of total intensity explained
            - "median_abs_ppm": median absolute ppm error
            - "mean_ppm_bias": mean signed ppm error (indicates systematic mass bias)
            
    Notes
    -----
    - Arrays are automatically sorted if needed
    - Greedy matching: each experimental peak matched to at most one theoretical ion
    - For overlapping windows, most intense (or closest) peak is selected
    
    Examples
    --------
    >>> exp_mz = np.array([100.1, 200.2, 300.3])
    >>> exp_int = np.array([1000, 2000, 500])
    >>> theo_mz = np.array([100.0, 200.0, 400.0])
    >>> result = match_theoretical_to_experimental(exp_mz, exp_int, theo_mz, ppm_tol=20)
    >>> result["mask"]
    array([ True,  True, False])
    >>> result["metrics"]["n_matched"]
    2
    >>> result["matched_annotation"]
    ['frag_0', 'frag_1', None]
    """
    # Validate input arrays
    if exp_mz.shape != exp_intensity.shape:
        raise ValueError(
            f"exp_mz shape {exp_mz.shape} must match exp_intensity shape {exp_intensity.shape}"
        )
    
    # Check for non-finite values
    if not np.all(np.isfinite(exp_mz)):
        raise ValueError("exp_mz contains non-finite values (NaN or inf)")
    if not np.all(np.isfinite(exp_intensity)):
        raise ValueError("exp_intensity contains non-finite values (NaN or inf)")
    if not np.all(np.isfinite(theo_mz)):
        raise ValueError("theo_mz contains non-finite values (NaN or inf)")
    
    # Validate annotations if provided
    if theo_annotations is not None and len(theo_annotations) != len(theo_mz):
        raise ValueError(
            f"theo_annotations length ({len(theo_annotations)}) must match "
            f"theo_mz length ({len(theo_mz)})"
        )
    
    # Handle empty arrays
    if exp_mz.size == 0 or theo_mz.size == 0:
        N = exp_mz.size
        return {
            "mask": np.zeros(N, dtype=bool),
            "match_idx": np.full(N, -1, dtype=np.int32),
            "ppm_error": np.full(N, np.nan, dtype=np.float64),
            "matched_annotation": [None] * N,
            "matched_theo_mz": np.full(N, np.nan, dtype=np.float64),
            "metrics": {
                "n_matched": 0,
                "frac_intensity": 0.0,
                "median_abs_ppm": np.nan,
                "mean_ppm_bias": np.nan,
            },
        }
    
    # Ensure arrays are sorted
    exp_mz, exp_intensity = _ensure_sorted(exp_mz, exp_intensity)
    theo_mz_result = _ensure_sorted(theo_mz)
    if isinstance(theo_mz_result, tuple):
        theo_mz = theo_mz_result[0]
    else:
        theo_mz = theo_mz_result
    
    # Calculate tolerance windows for each theoretical ion.
    # Da mode uses a fixed-width window; ppm mode scales with m/z.
    if da_tol is not None:
        tol_da = np.full(theo_mz.shape, da_tol, dtype=np.float64)
    else:
        tol_da = theo_mz * ppm_tol * 1e-6
        # Warn only in ppm mode when the window is suspiciously large, which
        # often indicates low-res CID data where da_tol should be used instead.
        max_tol_da = float(np.max(tol_da)) if tol_da.size > 0 else 0.0
        if max_tol_da > 1.0:
            warnings.warn(
                f"Large ppm tolerance window detected (max {max_tol_da:.3f} Da at "
                f"m/z {theo_mz[np.argmax(tol_da)]:.2f}). "
                "For low-resolution CID data consider using da_tol instead "
                "(e.g. da_tol=0.5).",
                UserWarning,
            )
    lo_bounds = np.searchsorted(exp_mz, theo_mz - tol_da, side="left")
    hi_bounds = np.searchsorted(exp_mz, theo_mz + tol_da, side="right")
    
    # Initialize output arrays
    N = exp_mz.size
    mask = np.zeros(N, dtype=bool)
    match_idx = np.full(N, -1, dtype=np.int32)
    ppm_error = np.full(N, np.nan, dtype=np.float64)
    matched_annotation = [None] * N
    matched_theo_mz = np.full(N, np.nan, dtype=np.float64)
    
    # Build all candidate matches within tolerance
    candidates = []
    for j, (lo, hi, mz_theo) in enumerate(zip(lo_bounds, hi_bounds, theo_mz)):
        if lo >= hi:
            continue  # No peaks in window
        
        for i in range(lo, hi):
            ppm_err = (exp_mz[i] - mz_theo) / mz_theo * 1e6
            # Verify candidate is within tolerance (defense against fp precision)
            if da_tol is not None:
                if abs(exp_mz[i] - mz_theo) > da_tol * 1.01:
                    continue
            else:
                if abs(ppm_err) > ppm_tol * 1.01:
                    continue
            
            if use_closest:
                # Use absolute ppm error as priority
                priority = abs(ppm_err)
            else:
                # Use negative intensity as priority (higher intensity = lower priority value)
                priority = -exp_intensity[i]
            
            candidates.append((i, j, priority, ppm_err))
    
    # Sort candidates by priority (closest ppm error or highest intensity)
    candidates.sort(key=lambda x: x[2])
    
    # Greedily assign matches (one-to-one)
    used_exp = set()
    used_theo = set()
    for i, j, _, ppm_err in candidates:
        if i not in used_exp and j not in used_theo:
            mask[i] = True
            match_idx[i] = j
            ppm_error[i] = ppm_err
            matched_theo_mz[i] = theo_mz[j]
            if theo_annotations is not None:
                matched_annotation[i] = theo_annotations[j]
            used_exp.add(i)
            used_theo.add(j)
    
    # Compute summary metrics
    matched_intensity = exp_intensity[mask].sum()
    total_intensity = exp_intensity.sum()
    frac_intensity = (
        float(matched_intensity / total_intensity) if total_intensity > 0 else 0.0
    )
    
    valid_errors = ppm_error[~np.isnan(ppm_error)]
    median_abs_ppm = float(np.median(np.abs(valid_errors))) if valid_errors.size > 0 else np.nan
    mean_ppm_bias = float(np.mean(valid_errors)) if valid_errors.size > 0 else np.nan
    
    return {
        "mask": mask,
        "match_idx": match_idx,
        "ppm_error": ppm_error,
        "matched_annotation": matched_annotation,
        "matched_theo_mz": matched_theo_mz,
        "metrics": {
            "n_matched": int(mask.sum()),
            "frac_intensity": frac_intensity,
            "median_abs_ppm": median_abs_ppm,
            "mean_ppm_bias": mean_ppm_bias,
        },
    }


def match_theoretical_to_experimental_rustyms(
    exp_mz: np.ndarray,
    exp_intensity: np.ndarray,
    sequence: str,
    precursor_mz: float,
    precursor_charge: int,
    fragmentation_type: Optional[str] = None,
    ppm_tol: float = 10.0,
    ion_types: Sequence[str] = ("b", "y"),
    max_charge: Optional[int] = None,
    use_direct_annotation: bool = False,
) -> Dict[str, Any]:
    """Match experimental spectrum to theoretical using rustyms/annotator.
    
    This function uses rustyms to generate theoretical fragments and then matches
    them using the standard matching algorithm. Optionally attempts to use rustyms'
    built-in annotation if MatchingParameters can be constructed.
    
    Parameters
    ----------
    exp_mz : np.ndarray
        Experimental m/z values
    exp_intensity : np.ndarray
        Experimental intensity values
    sequence : str
        Peptide sequence (ProForma format)
    precursor_mz : float
        Precursor m/z value
    precursor_charge : int
        Precursor charge state
    fragmentation_type : Optional[str], default=None
        Fragmentation type (e.g., "HCD", "CID", "ETD")
    ppm_tol : float, default=10.0
        Tolerance in parts per million
    ion_types : Sequence[str], default=("b", "y")
        Fragment ion types to generate
    max_charge : Optional[int], default=None
        Maximum fragment charge state
    use_direct_annotation : bool, default=False
        If True, attempt to use rustyms' RawSpectrum.annotate() directly.
        Currently not fully supported as MatchingParameters is not accessible
        from Python bindings.
        
    Returns
    -------
    Dict[str, Any]
        Same structure as match_theoretical_to_experimental() output
    """
    _check_rustyms_available("annotator")
    
    # Attempt direct annotation if requested (currently not fully supported)
    if use_direct_annotation:
        try:
            # Create RawSpectrum
            raw_spectrum = rustyms.RawSpectrum(
                title="annotation",
                num_scans=1,
                mz_array=exp_mz.tolist(),
                intensity_array=exp_intensity.tolist(),
                precursor_charge=precursor_charge,
                precursor_mass=(precursor_mz - PROTON_MASS) * precursor_charge,
            )
            
            # Create peptidoform
            peptidoform = rustyms.CompoundPeptidoformIon(sequence)
            
            # Map fragmentation model
            frag_model_name = _map_fragmentation_type_to_rustyms(fragmentation_type)
            frag_model_map = {
                "hcd": rustyms.FragmentationModel.CidHcd,
                "cid": rustyms.FragmentationModel.CidHcd,
                "etd": rustyms.FragmentationModel.Etd,
                "uvpd": rustyms.FragmentationModel.Uvpd,
            }
            frag_model = frag_model_map.get(frag_model_name, rustyms.FragmentationModel.CidHcd)
            
            # Try to annotate (requires MatchingParameters - not accessible from Python)
            # This will fail, but we catch and fall back to manual matching
            # TODO: If MatchingParameters becomes accessible, implement direct annotation
            warnings.warn(
                "Direct rustyms annotation not yet available. MatchingParameters "
                "is not accessible from Python bindings. Using manual matching."
            )
        except Exception:
            # Direct annotation failed, fall back to manual matching
            pass
    
    # Generate theoretical spectrum using rustyms
    theo_mz, theo_annotations = generate_theoretical_spectrum_rustyms(
        peptide=sequence,
        precursor_charge=precursor_charge,
        ion_types=ion_types,
        max_charge=max_charge,
        fragmentation_type=fragmentation_type,
    )
    
    # Use standard matching function (works with any theoretical spectrum)
    return match_theoretical_to_experimental(
        exp_mz=exp_mz,
        exp_intensity=exp_intensity,
        theo_mz=theo_mz,
        ppm_tol=ppm_tol,
        theo_annotations=theo_annotations,
    )


# =============================================================================
# 2b. Conditional Feature Annotation (Two-Pass Strategy)
# =============================================================================


def _extract_charge_from_annotation(annotation: str) -> int:
    """Extract charge state from annotation, handling both formats:

    - Fragment ions: ``'b3+'`` → 1, ``'y5++'`` → 2 (count ``+`` symbols)
    - Precursor ions (PSI mzPAF): ``'p^2'`` → 2, ``'p-H2O^3'`` → 3 (parse ``^z``)

    Parameters
    ----------
    annotation : str
        Ion annotation string.

    Returns
    -------
    int
        Charge state (≥ 1).
    """
    # PSI mzPAF '^z' notation — precursor ions
    caret_idx = annotation.rfind('^')
    if caret_idx != -1:
        charge_str = ''
        for ch in annotation[caret_idx + 1:]:
            if ch.isdigit():
                charge_str += ch
            else:
                break
        if charge_str:
            return int(charge_str)
    # Fragment ion format — count '+' symbols
    count = annotation.count('+')
    return max(1, count)


def _generate_neutral_loss_variants(
    mz: float,
    annotation: str,
    charge: int,
    loss_types: Sequence[str] = ("H2O", "NH3"),
) -> Tuple[List[float], List[str]]:
    """Generate neutral loss variants for a matched fragment ion.
    
    Parameters
    ----------
    mz : float
        m/z value of parent ion
    annotation : str
        Parent ion annotation
    charge : int
        Charge state
    loss_types : Sequence[str], default=("H2O", "NH3")
        Types of neutral losses to generate
        
    Returns
    -------
    Tuple[List[float], List[str]]
        (loss_mz_values, loss_annotations)
    """
    # Neutral loss masses (Da)
    LOSS_MASSES = {
        "H2O": 18.01056,   # Water
        "NH3": 17.02655,   # Ammonia
        "CO": 27.99491,    # Carbon monoxide (for a-ions from b-ions)
        "H3PO4": 97.97690, # Phosphoric acid (phosphorylation)
    }
    
    loss_mz = []
    loss_ann = []
    
    for loss_name in loss_types:
        if loss_name not in LOSS_MASSES:
            warnings.warn(f"Unknown neutral loss type: {loss_name}", stacklevel=2)
            continue
        
        loss_mass = LOSS_MASSES[loss_name]
        loss_mz_val = mz - (loss_mass / charge)
        loss_mz.append(loss_mz_val)
        loss_ann.append(f"{annotation}-{loss_name}")
    
    return loss_mz, loss_ann


def _generate_isotope_variants(
    mz: float,
    annotation: str,
    charge: int,
    max_isotope: int = 3,
) -> Tuple[List[float], List[str]]:
    """Generate isotopic variants for a matched fragment ion.
    
    Parameters
    ----------
    mz : float
        m/z value of monoisotopic peak
    annotation : str
        Monoisotopic peak annotation
    charge : int
        Charge state
    max_isotope : int, default=3
        Maximum isotope number to generate (+1, +2, +3, etc.)
        
    Returns
    -------
    Tuple[List[float], List[str]]
        (isotope_mz_values, isotope_annotations)
    """
    ISOTOPE_MASS_DIFF = 1.003355  # 13C - 12C mass difference (Da)
    
    iso_mz = []
    iso_ann = []
    
    for iso_num in range(1, max_isotope + 1):
        iso_mz_val = mz + (ISOTOPE_MASS_DIFF * iso_num / charge)
        iso_mz.append(iso_mz_val)
        iso_ann.append(f"{annotation}[+{iso_num}]")
    
    return iso_mz, iso_ann


def _generate_precursor_ions(
    aa_seq: "AASequence",
    precursor_charge: int,
    add_losses: bool = False,
    loss_types: Sequence[str] = ("H2O", "NH3"),
    add_isotopes: bool = False,
    max_isotope: int = 3,
) -> Tuple[List[float], List[str]]:
    """Generate precursor-related ions using PSI mzPAF annotation format.

    Generates the intact precursor ion [M+zH]^z+ and, if *add_losses* is True,
    precursor neutral loss variants from *loss_types*.  When *add_isotopes* is
    True, M+1 through M+*max_isotope* peaks are appended for every base
    precursor entry (intact + each neutral loss).

    Annotation format follows the HUPO PSI mzPAF standard:

    - Intact precursor:  ``p^{z}``   (e.g. ``p^2``)
    - Neutral loss:       ``p-{loss}^{z}``  (e.g. ``p-H2O^2``)
    - Isotope:            ``p^{z}[+{n}]``   (e.g. ``p^2[+1]``)

    Parameters
    ----------
    aa_seq : AASequence
        Parsed PyOpenMS peptide sequence object.
    precursor_charge : int
        Precursor charge state *z*.
    add_losses : bool, default=False
        Generate precursor neutral loss variants.
    loss_types : Sequence[str], default=("H2O", "NH3")
        Neutral loss types. Supported: ``"H2O"``, ``"NH3"``, ``"H3PO4"``, ``"SO3"``.
        Unknown values are silently skipped.
    add_isotopes : bool, default=False
        Append isotopic variants (M+1 … M+max_isotope) for each precursor entry.
    max_isotope : int, default=3
        Maximum isotope offset when *add_isotopes* is True.

    Returns
    -------
    Tuple[List[float], List[str]]
        *(mz_values, annotations)* in PSI mzPAF format.
    """
    PRECURSOR_LOSS_MASSES: Dict[str, float] = {
        "H2O":   18.010565,   # water
        "NH3":   17.026549,   # ammonia
        "H3PO4": 97.976895,   # phosphoric acid (phosphopeptides)
        "SO3":   79.956815,   # sulfur trioxide (sulfation)
    }

    z = precursor_charge
    neutral_mass = aa_seq.getMonoWeight()  # monoisotopic neutral peptide mass
    intact_mz = (neutral_mass + z * PROTON_MASS) / z

    mz_out: List[float] = [intact_mz]
    ann_out: List[str] = [f"p^{z}"]

    if add_losses:
        for loss_name in loss_types:
            loss_mass = PRECURSOR_LOSS_MASSES.get(loss_name)
            if loss_mass is None:
                continue
            loss_mz = (neutral_mass + z * PROTON_MASS - loss_mass) / z
            if loss_mz > 0:
                mz_out.append(loss_mz)
                ann_out.append(f"p-{loss_name}^{z}")

    # Append isotopic variants for every base precursor entry
    if add_isotopes:
        ISOTOPE_MASS_DIFF = 1.003355  # 13C - 12C mass difference (Da)
        base_mz = list(mz_out)  # snapshot before appending
        base_ann = list(ann_out)
        for mz_val, ann_val in zip(base_mz, base_ann):
            for iso_num in range(1, max_isotope + 1):
                mz_out.append(mz_val + ISOTOPE_MASS_DIFF * iso_num / z)
                ann_out.append(f"{ann_val}[+{iso_num}]")

    return mz_out, ann_out


def match_with_conditional_features(
    exp_mz: np.ndarray,
    exp_intensity: np.ndarray,
    peptide: str,
    precursor_charge: int = 2,
    ppm_tol: float = 10.0,
    da_tol: Optional[float] = None,
    ion_types: Sequence[str] = ("b", "y"),
    max_charge: Optional[int] = None,
    add_losses: bool = True,
    loss_types: Sequence[str] = ("H2O", "NH3"),
    add_isotopes: bool = True,
    max_isotope: int = 3,
    isotope_intensity_threshold: float = 0.01,
    add_precursor: bool = False,
    use_closest: bool = False,
    engine: str = "pyopenms",
    fragmentation_type: Optional[str] = None,
    _fast_mode: bool = True,
) -> Dict[str, Any]:
    """Match experimental spectrum using two-pass conditional annotation strategy.

    This function reduces false positives by using a two-pass approach:

    **Pass 1**: Match base fragment ions (b, y, etc.) at the fragment tolerance
    and precursor ions (including unconditional isotopes M+1…M+max_isotope)
    at ``2×`` the fragment tolerance to accommodate systematic instrument mass
    offsets.

    **Pass 2**: For each matched base *fragment* ion, conditionally check for:
    - Neutral losses (if parent ion matched)
    - Isotopes (if monoisotopic peak matched and intense enough)
    Precursor isotopes are skipped in Pass 2 (already in Pass 1).

    This approach dramatically reduces false positives because:
    - Neutral loss peaks cannot exist without their parent ion
    - Isotope peaks are only checked for intense monoisotopic peaks
    - Far fewer theoretical peaks need to be generated and matched

    Parameters
    ----------
    exp_mz : np.ndarray
        Experimental m/z values
    exp_intensity : np.ndarray
        Experimental intensity values
    peptide : str
        Peptide sequence
    precursor_charge : int, default=2
        Precursor charge state
    ppm_tol : float, default=10.0
        Tolerance in parts per million.  Ignored when ``da_tol`` is set.
    da_tol : Optional[float], default=None
        Absolute tolerance in Daltons.  When provided, overrides ``ppm_tol``.
        Recommended for low-resolution CID (ion-trap MS2) data; start with
        0.5 Da.  Precursor ions are matched at ``2 × da_tol``.
    ion_types : Sequence[str], default=("b", "y")
        Fragment ion types to generate
    max_charge : Optional[int], default=None
        Maximum fragment charge state
    add_losses : bool, default=True
        Check for neutral losses from matched base ions
    loss_types : Sequence[str], default=("H2O", "NH3")
        Types of neutral losses to check for
    add_isotopes : bool, default=True
        Check for isotopes from matched base ions
    max_isotope : int, default=3
        Maximum isotope number (+1, +2, +3, etc.)
    isotope_intensity_threshold : float, default=0.01
        Minimum relative intensity (fraction of base peak) for checking isotopes.
        Only matched peaks above this threshold will have isotopes checked.
    add_precursor : bool, default=False
        Include precursor ion in theoretical spectrum
    use_closest : bool, default=False
        Use closest peak instead of most intense when matching
    engine : str, default="pyopenms"
        Engine to use: "pyopenms" or "annotator"/"rustyms"
    fragmentation_type : Optional[str], default=None
        Fragmentation type (e.g., "HCD", "CID", "ETD")
    _fast_mode : bool, default=True
        Internal parameter for optimization. When True, uses optimized matching
        that reduces overhead from multiple passes.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "mask": boolean array marking matched experimental peaks
        - "match_idx": index of matched theoretical ion (-1 if unmatched)
        - "ppm_error": ppm error for each experimental peak (NaN if unmatched)
        - "matched_annotation": fragment annotation for each peak (None if unmatched)
        - "matched_theo_mz": theoretical m/z for each peak (NaN if unmatched)
        - "feature_type": type of feature ("base", "loss", "isotope", "precursor", None)
        - "parent_annotation": for losses/isotopes, the parent ion annotation
        - "metrics": dict with summary statistics:
            - "n_matched": total number of matched peaks
            - "n_base": number of matched base fragment ions
            - "n_precursor": number of matched precursor ions
            - "n_losses": number of matched neutral losses
            - "n_isotopes": number of matched isotopes
            - "frac_intensity": fraction of total intensity explained
            - "median_abs_ppm": median absolute ppm error
            - "mean_ppm_bias": mean signed ppm error
            
    Examples
    --------
    >>> result = match_with_conditional_features(
    ...     exp_mz, exp_intensity, "PEPTIDE",
    ...     ppm_tol=10.0,
    ...     add_losses=True,
    ...     add_isotopes=True,
    ...     isotope_intensity_threshold=0.02  # Only check isotopes for peaks >2% base peak
    ... )
    >>> result["metrics"]["n_base"]  # Number of base fragment ions matched
    12
    >>> result["metrics"]["n_losses"]  # Number of neutral losses matched
    3
    >>> result["feature_type"]  # Type of each matched peak
    ['base', 'loss', 'isotope', None, 'base', ...]
    """
    # Validate inputs
    if exp_mz.shape != exp_intensity.shape:
        raise ValueError("exp_mz and exp_intensity must have same shape")
    
    if exp_mz.size == 0:
        return {
            "mask": np.array([], dtype=bool),
            "match_idx": np.array([], dtype=np.int32),
            "ppm_error": np.array([], dtype=np.float64),
            "matched_annotation": [],
            "matched_theo_mz": np.array([], dtype=np.float64),
            "feature_type": [],
            "parent_annotation": [],
            "metrics": {
                "n_matched": 0,
                "n_base": 0,
                "n_precursor": 0,
                "n_losses": 0,
                "n_isotopes": 0,
                "frac_intensity": 0.0,
                "median_abs_ppm": np.nan,
                "mean_ppm_bias": np.nan,
            },
        }
    
    # Ensure sorted
    exp_mz, exp_intensity = _ensure_sorted(exp_mz, exp_intensity)
    
    # Calculate base peak intensity for isotope threshold
    base_peak_intensity = exp_intensity.max()
    
    # =============================================================================
    # PASS 1: Match base fragment ions + unconditional precursor isotopes
    # =============================================================================

    # Generate theoretical spectrum with precursor isotopes included
    # unconditionally (large peptides often lack the monoisotopic peak).
    theo_mz, theo_ann = generate_theoretical_spectrum(
        peptide=peptide,
        precursor_charge=precursor_charge,
        ion_types=ion_types,
        max_charge=max_charge,
        add_losses=False,  # Don't generate losses yet
        add_isotopes=False,  # Don't generate isotopes yet
        add_precursor=add_precursor,
        precursor_isotopes=add_precursor,  # unconditional precursor isotopes
        max_precursor_isotope=max_isotope,
        engine=engine,
        fragmentation_type=fragmentation_type,
    )

    # Split theoretical peaks into fragment and precursor groups so that
    # precursor ions can be matched with a wider tolerance (2× ppm_tol) to
    # accommodate systematic mass offsets in instrument-reported precursor m/z.
    _is_precursor_theo = np.array(
        [a.startswith("p^") or a.startswith("p-") for a in theo_ann],
        dtype=bool,
    )
    frag_idx = np.where(~_is_precursor_theo)[0]
    prec_idx = np.where(_is_precursor_theo)[0]

    N = exp_mz.size

    # --- Match fragment ions at normal tolerance ---
    if frag_idx.size > 0:
        frag_theo_mz = theo_mz[frag_idx]
        frag_theo_ann = [theo_ann[j] for j in frag_idx]
        frag_matches = match_theoretical_to_experimental(
            exp_mz=exp_mz,
            exp_intensity=exp_intensity,
            theo_mz=frag_theo_mz,
            ppm_tol=ppm_tol,
            da_tol=da_tol,
            use_closest=use_closest,
            theo_annotations=frag_theo_ann,
        )
    else:
        frag_matches = {
            "mask": np.zeros(N, dtype=bool),
            "match_idx": np.full(N, -1, dtype=np.int32),
            "ppm_error": np.full(N, np.nan, dtype=np.float64),
            "matched_annotation": [None] * N,
            "matched_theo_mz": np.full(N, np.nan, dtype=np.float64),
        }

    # --- Match precursor ions at 2× tolerance ---
    if prec_idx.size > 0:
        prec_theo_mz = theo_mz[prec_idx]
        prec_theo_ann = [theo_ann[j] for j in prec_idx]
        prec_matches = match_theoretical_to_experimental(
            exp_mz=exp_mz,
            exp_intensity=exp_intensity,
            theo_mz=prec_theo_mz,
            ppm_tol=ppm_tol * 2,
            da_tol=da_tol * 2 if da_tol is not None else None,
            use_closest=use_closest,
            theo_annotations=prec_theo_ann,
        )
    else:
        prec_matches = {
            "mask": np.zeros(N, dtype=bool),
            "match_idx": np.full(N, -1, dtype=np.int32),
            "ppm_error": np.full(N, np.nan, dtype=np.float64),
            "matched_annotation": [None] * N,
            "matched_theo_mz": np.full(N, np.nan, dtype=np.float64),
        }

    # --- Merge fragment + precursor matches (fragment wins on conflict) ---
    final_mask = frag_matches["mask"].copy()
    final_match_idx = frag_matches["match_idx"].copy()
    final_ppm_error = frag_matches["ppm_error"].copy()
    final_matched_annotation = frag_matches["matched_annotation"].copy()
    final_matched_theo_mz = frag_matches["matched_theo_mz"].copy()

    # Remap fragment match_idx back to theo_mz indices
    for i in range(N):
        if final_mask[i]:
            final_match_idx[i] = int(frag_idx[final_match_idx[i]])

    # Fill in precursor matches where fragment didn't match
    for i in range(N):
        if prec_matches["mask"][i] and not final_mask[i]:
            final_mask[i] = True
            final_match_idx[i] = int(prec_idx[prec_matches["match_idx"][i]])
            final_ppm_error[i] = prec_matches["ppm_error"][i]
            final_matched_annotation[i] = prec_matches["matched_annotation"][i]
            final_matched_theo_mz[i] = prec_matches["matched_theo_mz"][i]

    # Mark feature types
    feature_type = []
    for i in range(N):
        if not final_mask[i]:
            feature_type.append(None)
        else:
            ann = final_matched_annotation[i]
            if ann and (ann.startswith("p^") or ann.startswith("p-")):
                # Precursor isotope peaks get "isotope" type with parent tracking
                if "[+" in ann:
                    feature_type.append("isotope")
                else:
                    feature_type.append("precursor")
            else:
                feature_type.append("base")

    parent_annotation = [None] * N
    # Set parent annotations for precursor isotope peaks
    for i in range(N):
        if feature_type[i] == "isotope":
            ann = final_matched_annotation[i]
            if ann and "[+" in ann:
                parent_annotation[i] = ann.split("[")[0]

    # Count base ions, precursor ions, and precursor isotopes separately
    n_base = sum(1 for ft in feature_type if ft == "base")
    n_precursor = sum(1 for ft in feature_type if ft == "precursor")
    n_losses = 0
    n_isotopes = sum(1 for ft in feature_type if ft == "isotope")
    
    # =============================================================================
    # PASS 2: For each matched base ion, check for losses and isotopes
    # =============================================================================
    
    # Early exit if no losses or isotopes requested
    if not add_losses and not add_isotopes:
        _matched_int = exp_intensity[final_mask].sum()
        _total_int = exp_intensity.sum()
        _frac = float(_matched_int / _total_int) if _total_int > 0 else 0.0
        _valid = final_ppm_error[~np.isnan(final_ppm_error)]
        return {
            "mask": final_mask,
            "match_idx": final_match_idx,
            "ppm_error": final_ppm_error,
            "matched_annotation": final_matched_annotation,
            "matched_theo_mz": final_matched_theo_mz,
            "feature_type": feature_type,
            "parent_annotation": parent_annotation,
            "metrics": {
                "n_matched": int(final_mask.sum()),
                "n_base": n_base,
                "n_precursor": n_precursor,
                "n_losses": 0,
                "n_isotopes": n_isotopes,
                "frac_intensity": _frac,
                "median_abs_ppm": float(np.median(np.abs(_valid))) if _valid.size > 0 else np.nan,
                "mean_ppm_bias": float(np.mean(_valid)) if _valid.size > 0 else np.nan,
            },
        }
    
    # Build list of conditional features to check
    # Pre-allocate lists with estimated size for efficiency
    max_conditional = n_base * (len(loss_types) if add_losses else 0 + max_isotope if add_isotopes else 0)
    conditional_mz = []
    conditional_ann = []
    conditional_feature_types = []
    conditional_parent_ann = []
    
    # Pre-compute loss masses to avoid dictionary lookups in loop
    LOSS_MASSES = {
        "H2O": 18.01056,
        "NH3": 17.02655,
        "CO": 27.99491,
        "H3PO4": 97.97690,
        "SO3": 79.95682,
    }
    loss_masses_list = [(name, LOSS_MASSES.get(name, 0.0)) for name in loss_types if name in LOSS_MASSES]
    
    ISOTOPE_MASS_DIFF = 1.003355
    
    for i in range(N):
        if not final_mask[i]:
            continue  # Skip unmatched peaks

        # Only generate conditional features for base fragment or precursor ions
        # (not for precursor isotopes already matched in Pass 1)
        ft = feature_type[i]
        if ft not in ("base", "precursor"):
            continue

        matched_idx = final_match_idx[i]
        base_mz_val = theo_mz[matched_idx]
        base_ann = theo_ann[matched_idx]
        exp_int = exp_intensity[i]

        is_precursor = (ft == "precursor")

        # Extract charge — delegates to the unified helper (handles both p^z and b/y++ formats)
        charge = _extract_charge_from_annotation(base_ann)

        # Generate neutral losses conditionally on matched parent ion
        if add_losses:
            if is_precursor:
                # Precursor neutral losses (conditional on intact precursor matched)
                for loss_name, loss_mass in loss_masses_list:
                    loss_mz_val = base_mz_val - (loss_mass / charge)
                    loss_ann = f"p-{loss_name}^{charge}"
                    conditional_mz.append(loss_mz_val)
                    conditional_ann.append(loss_ann)
                    conditional_feature_types.append("loss")
                    conditional_parent_ann.append(base_ann)
                    # Isotope variants of this precursor loss
                    if add_isotopes:
                        for iso_num in range(1, max_isotope + 1):
                            conditional_mz.append(
                                loss_mz_val + ISOTOPE_MASS_DIFF * iso_num / charge
                            )
                            conditional_ann.append(f"{loss_ann}[+{iso_num}]")
                            conditional_feature_types.append("isotope")
                            conditional_parent_ann.append(loss_ann)
            else:
                # Fragment ion neutral losses (existing behaviour)
                for loss_name, loss_mass in loss_masses_list:
                    loss_mz_val = base_mz_val - (loss_mass / charge)
                    conditional_mz.append(loss_mz_val)
                    conditional_ann.append(f"{base_ann}-{loss_name}")
                    conditional_feature_types.append("loss")
                    conditional_parent_ann.append(base_ann)

        # Generate isotopes for fragment ions only (precursor isotopes were
        # already generated unconditionally in Pass 1).
        if add_isotopes and not is_precursor:
            relative_intensity = exp_int / base_peak_intensity
            if relative_intensity >= isotope_intensity_threshold:
                for iso_num in range(1, max_isotope + 1):
                    iso_mz_val = base_mz_val + (ISOTOPE_MASS_DIFF * iso_num / charge)
                    conditional_mz.append(iso_mz_val)
                    conditional_ann.append(f"{base_ann}[+{iso_num}]")
                    conditional_feature_types.append("isotope")
                    conditional_parent_ann.append(base_ann)
    
    # Match conditional features if any were generated.
    # Split into fragment vs precursor groups so precursor-related conditionals
    # (neutral losses) use the wider 2× tolerance (same systematic offset as
    # the precursor itself).
    if conditional_mz:
        conditional_theo_mz = np.array(conditional_mz, dtype=np.float64)

        # Sort conditional features by m/z for efficient matching
        sort_idx = np.argsort(conditional_theo_mz)
        conditional_theo_mz = conditional_theo_mz[sort_idx]
        conditional_ann = [conditional_ann[i] for i in sort_idx]
        conditional_feature_types = [conditional_feature_types[i] for i in sort_idx]
        conditional_parent_ann = [conditional_parent_ann[i] for i in sort_idx]

        # Separate precursor-related conditionals from fragment conditionals
        _is_prec_cond = np.array(
            [a.startswith("p^") or a.startswith("p-") for a in conditional_ann],
            dtype=bool,
        )
        cond_frag_idx = np.where(~_is_prec_cond)[0]
        cond_prec_idx = np.where(_is_prec_cond)[0]

        # Helper to merge a set of conditional matches into the final arrays
        def _merge_conditional(matches, c_ann, c_theo_mz, c_ftypes, c_parent, idx_map):
            nonlocal n_losses, n_isotopes
            for i in range(N):
                if matches["mask"][i] and not final_mask[i]:
                    cond_idx = matches["match_idx"][i]
                    orig_idx = int(idx_map[cond_idx])
                    final_mask[i] = True
                    final_match_idx[i] = len(theo_mz) + orig_idx
                    final_ppm_error[i] = matches["ppm_error"][i]
                    final_matched_annotation[i] = c_ann[cond_idx]
                    final_matched_theo_mz[i] = c_theo_mz[cond_idx]
                    ftype = c_ftypes[cond_idx]
                    feature_type[i] = ftype
                    parent_annotation[i] = c_parent[cond_idx]
                    if ftype == "loss":
                        n_losses += 1
                    elif ftype == "isotope":
                        n_isotopes += 1

        # Match fragment conditionals at normal tolerance
        if cond_frag_idx.size > 0:
            frag_c_mz = conditional_theo_mz[cond_frag_idx]
            frag_c_ann = [conditional_ann[j] for j in cond_frag_idx]
            frag_c_ft = [conditional_feature_types[j] for j in cond_frag_idx]
            frag_c_pa = [conditional_parent_ann[j] for j in cond_frag_idx]
            frag_c_matches = match_theoretical_to_experimental(
                exp_mz=exp_mz,
                exp_intensity=exp_intensity,
                theo_mz=frag_c_mz,
                ppm_tol=ppm_tol,
                da_tol=da_tol,
                use_closest=use_closest,
                theo_annotations=frag_c_ann,
            )
            _merge_conditional(frag_c_matches, frag_c_ann, frag_c_mz,
                               frag_c_ft, frag_c_pa, cond_frag_idx)

        # Match precursor conditionals at 2× tolerance
        if cond_prec_idx.size > 0:
            prec_c_mz = conditional_theo_mz[cond_prec_idx]
            prec_c_ann = [conditional_ann[j] for j in cond_prec_idx]
            prec_c_ft = [conditional_feature_types[j] for j in cond_prec_idx]
            prec_c_pa = [conditional_parent_ann[j] for j in cond_prec_idx]
            prec_c_matches = match_theoretical_to_experimental(
                exp_mz=exp_mz,
                exp_intensity=exp_intensity,
                theo_mz=prec_c_mz,
                ppm_tol=ppm_tol * 2,
                da_tol=da_tol * 2 if da_tol is not None else None,
                use_closest=use_closest,
                theo_annotations=prec_c_ann,
            )
            _merge_conditional(prec_c_matches, prec_c_ann, prec_c_mz,
                               prec_c_ft, prec_c_pa, cond_prec_idx)
    
    # Compute final metrics
    matched_intensity = exp_intensity[final_mask].sum()
    total_intensity = exp_intensity.sum()
    frac_intensity = (
        float(matched_intensity / total_intensity) if total_intensity > 0 else 0.0
    )
    
    valid_errors = final_ppm_error[~np.isnan(final_ppm_error)]
    median_abs_ppm = float(np.median(np.abs(valid_errors))) if valid_errors.size > 0 else np.nan
    mean_ppm_bias = float(np.mean(valid_errors)) if valid_errors.size > 0 else np.nan
    
    return {
        "mask": final_mask,
        "match_idx": final_match_idx,
        "ppm_error": final_ppm_error,
        "matched_annotation": final_matched_annotation,
        "matched_theo_mz": final_matched_theo_mz,
        "feature_type": feature_type,
        "parent_annotation": parent_annotation,
        "metrics": {
            "n_matched": int(final_mask.sum()),
            "n_base": n_base,
            "n_precursor": n_precursor,
            "n_losses": n_losses,
            "n_isotopes": n_isotopes,
            "frac_intensity": frac_intensity,
            "median_abs_ppm": median_abs_ppm,
            "mean_ppm_bias": mean_ppm_bias,
        },
    }


# =============================================================================
# 3. Custom Ion Detection
# =============================================================================


def detect_custom_ions(
    exp_mz: np.ndarray,
    exp_intensity: np.ndarray,
    custom_ions: Optional[Dict[str, List[float]]] = None,
    ppm_tol: float = 10.0,
    return_details: bool = False,
    add_isotopes: bool = False,
    max_isotope: int = 2,
    isotope_intensity_threshold: float = 0.02,
) -> Dict[str, Dict[str, Any]]:
    """Detect presence of custom ion types in experimental spectrum.

    Uses a conditional two-pass strategy:

    - **Pass 1** (monoisotopic): scan for each target m/z in *custom_ions*.
    - **Pass 2** (isotopes, optional): for every *detected* ion whose matched
      intensity exceeds *isotope_intensity_threshold* relative to the base peak,
      check for M+1 … M+*max_isotope* peaks at +1.003355 × n Da spacing
      (assumes z = 1, which holds for custom diagnostic ions in positive-mode
      MS2).

    Parameters
    ----------
    exp_mz : np.ndarray
        Experimental m/z values.
    exp_intensity : np.ndarray
        Experimental intensity values.
    custom_ions : Optional[Dict[str, List[float]]], default=None
        Dictionary mapping ion group names to target m/z values.
        If None, uses DEFAULT_CUSTOM_IONS.
    ppm_tol : float, default=10.0
        Tolerance in ppm for matching.
    return_details : bool, default=False
        If True, return detailed match information (matched m/z, intensities,
        and per-isotope match records).
    add_isotopes : bool, default=False
        Enable Pass 2 isotope checking for detected custom ions.
    max_isotope : int, default=2
        Highest isotope offset to check (M+1 … M+max_isotope).
    isotope_intensity_threshold : float, default=0.02
        Minimum relative intensity (vs spectrum base peak) a monoisotopic
        match must have to trigger isotope checking.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        For each ion group:
        - ``found``: bool
        - ``n_matched``: int (monoisotopic matches)
        - ``max_intensity``: float
        - ``matched_mz``: List[float]  (if *return_details*)
        - ``matched_intensity``: List[float]  (if *return_details*)
        - ``n_isotope_matches``: int  (if *add_isotopes*)
        - ``isotope_matches``: list of dicts  (if *add_isotopes* and *return_details*)
          Each dict: ``{"parent_mz", "isotope_num", "matched_mz", "matched_intensity"}``
    """
    NEUTRON_MASS = 1.003355  # 13C – 12C mass difference

    if custom_ions is None:
        custom_ions = DEFAULT_CUSTOM_IONS

    # Handle empty spectrum
    if exp_mz.size == 0:
        empty: Dict[str, Dict[str, Any]] = {}
        for group in custom_ions:
            entry: Dict[str, Any] = {
                "found": False,
                "n_matched": 0,
                "max_intensity": 0.0,
            }
            if return_details:
                entry["matched_mz"] = []
                entry["matched_intensity"] = []
            if add_isotopes:
                entry["n_isotope_matches"] = 0
                if return_details:
                    entry["isotope_matches"] = []
            empty[group] = entry
        return empty

    # Ensure sorted
    exp_mz, exp_intensity = _ensure_sorted(exp_mz, exp_intensity)
    base_peak_int = float(exp_intensity.max()) if exp_intensity.size > 0 else 1.0

    results: Dict[str, Dict[str, Any]] = {}

    # ── Pass 1: monoisotopic detection ──────────────────────────────────
    for group_name, target_mz_list in custom_ions.items():
        matched_mz: List[float] = []
        matched_intensity: List[float] = []
        max_int = 0.0

        for target_mz in target_mz_list:
            lo, hi = _ppm_window(float(target_mz), ppm_tol)

            i_lo = np.searchsorted(exp_mz, lo, side="left")
            i_hi = np.searchsorted(exp_mz, hi, side="right")

            if i_hi > i_lo:
                window_int = exp_intensity[i_lo:i_hi]
                window_mz = exp_mz[i_lo:i_hi]
                best_idx = int(window_int.argmax())

                matched_mz.append(float(window_mz[best_idx]))
                matched_intensity.append(float(window_int[best_idx]))
                max_int = max(max_int, float(window_int[best_idx]))

        results[group_name] = {
            "found": len(matched_mz) > 0,
            "n_matched": len(matched_mz),
            "max_intensity": max_int,
        }

        if return_details:
            results[group_name]["matched_mz"] = matched_mz
            results[group_name]["matched_intensity"] = matched_intensity

    # ── Pass 2: conditional isotope checking ────────────────────────────
    if add_isotopes:
        for group_name, group_data in results.items():
            isotope_matches: List[Dict[str, Any]] = []

            if group_data["found"]:
                mono_mz_list = group_data.get("matched_mz", [])
                mono_int_list = group_data.get("matched_intensity", [])

                for mono_mz, mono_int in zip(mono_mz_list, mono_int_list):
                    # Only check isotopes for sufficiently intense detections
                    if mono_int / base_peak_int < isotope_intensity_threshold:
                        continue

                    for n in range(1, max_isotope + 1):
                        iso_target = mono_mz + n * NEUTRON_MASS  # z=1 assumed
                        lo, hi = _ppm_window(iso_target, ppm_tol)

                        i_lo = np.searchsorted(exp_mz, lo, side="left")
                        i_hi = np.searchsorted(exp_mz, hi, side="right")

                        if i_hi > i_lo:
                            window_int = exp_intensity[i_lo:i_hi]
                            window_mz = exp_mz[i_lo:i_hi]
                            best_idx = int(window_int.argmax())

                            isotope_matches.append({
                                "parent_mz": mono_mz,
                                "isotope_num": n,
                                "matched_mz": float(window_mz[best_idx]),
                                "matched_intensity": float(window_int[best_idx]),
                            })

            group_data["n_isotope_matches"] = len(isotope_matches)
            if return_details:
                group_data["isotope_matches"] = isotope_matches

    return results


# =============================================================================
# 4. Bulk DataFrame Annotation
# =============================================================================


def annotate_dataframe(
    df: pl.DataFrame,
    *,
    ppm_tol: float = 10.0,
    da_tol: Optional[float] = None,
    ion_types: Sequence[str] = ("b", "y"),
    max_charge: Optional[int] = None,
    add_losses: bool = False,
    add_isotopes: bool = False,
    max_isotope: int = 3,
    isotope_model: Literal["coarse", "fine"] = "coarse",
    max_isotope_probability: float = 0.95,
    add_precursor: bool = False,
    detect_custom: bool = False,
    custom_ions: Optional[Dict[str, List[float]]] = None,
    progress: bool = True,
    use_closest_match: bool = False,
    engine: str = "pyopenms",
    fragmentation_type: Optional[str] = None,
    use_conditional_annotation: bool = True,
    loss_types: Sequence[str] = ("H2O", "NH3"),
    isotope_intensity_threshold: float = 0.01,
) -> pl.DataFrame:
    """Annotate DataFrame of spectra with theoretical fragment matching results.
    
    For each spectrum in the DataFrame, generates theoretical fragments and
    matches them to experimental peaks. Optionally detects custom ion types.
    Results are added as new columns.
    
    **Two Annotation Modes:**
    
    1. **Conditional (default, use_conditional_annotation=True)**:
       Two-pass strategy that reduces false positives:
       - Pass 1: Match base fragment ions only
       - Pass 2: For matched ions, check for neutral losses and isotopes
       This is recommended as it dramatically reduces false positives.
    
    2. **Traditional (use_conditional_annotation=False)**:
       Generates all theoretical features upfront and matches them.
       May produce more false positives but is simpler.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with required columns:
        - "sequence": peptide sequences
        - "precursor_charge": charge states
        - "mz_array": experimental m/z arrays
        - "intensity_array": experimental intensity arrays
    ppm_tol : float, default=10.0
        Tolerance in ppm for matching
    ion_types : Sequence[str], default=("b", "y")
        Fragment ion types to generate
    max_charge : Optional[int], default=None
        Maximum fragment charge
    add_losses : bool, default=False
        Include neutral losses
    add_isotopes : bool, default=False
        Include isotopic peaks
    max_isotope : int, default=3
        Maximum isotope number (for coarse model or conditional mode)
    isotope_model : {"coarse", "fine"}, default="coarse"
        Isotope model to use (only for traditional mode)
    max_isotope_probability : float, default=0.95
        Cumulative probability for fine isotope model (only for traditional mode)
    add_precursor : bool, default=False
        Include precursor ion in theoretical spectrum
    detect_custom : bool, default=False
        Whether to detect custom ions in experimental spectra
    custom_ions : Optional[Dict[str, List[float]]], default=None
        Dictionary of custom ions. If None, uses DEFAULT_CUSTOM_IONS.
    progress : bool, default=True
        Show progress bar
    use_closest_match : bool, default=False
        Use closest peak instead of most intense when matching
    engine : str, default="pyopenms"
        Engine to use: "pyopenms" or "annotator"/"rustyms"
    fragmentation_type : Optional[str], default=None
        Fragmentation type (e.g., "HCD", "CID", "ETD") - used by rustyms engine
    use_conditional_annotation : bool, default=True
        Use two-pass conditional annotation strategy (recommended).
        If False, uses traditional approach with all features generated upfront.
    loss_types : Sequence[str], default=("H2O", "NH3")
        Types of neutral losses to check (only for conditional mode)
    isotope_intensity_threshold : float, default=0.01
        Minimum relative intensity for checking isotopes (only for conditional mode).
        Peaks below this fraction of base peak intensity won't have isotopes checked.
        
    Returns
    -------
    pl.DataFrame
        Input DataFrame with added columns:
        - "signal_mask": boolean arrays marking matched peaks
        - "ppm_error": ppm errors for each peak
        - "matched_annotation": fragment ion annotations for each peak (None if unmatched)
        - "matched_theo_mz": theoretical m/z values for each peak (NaN if unmatched)
        - "n_matched": number of matched theoretical ions
        - "frac_intensity": fraction of intensity explained by matches
        - "median_abs_ppm": median absolute ppm error
        - "mean_ppm_bias": mean signed ppm error (systematic mass bias)
        - "feature_type": type of matched feature ("base", "loss", "isotope", None) (conditional mode only)
        - "parent_annotation": parent ion for losses/isotopes (conditional mode only)
        - "n_base": number of matched base ions (conditional mode only)
        - "n_losses": number of matched neutral losses (conditional mode only)
        - "n_isotopes": number of matched isotopes (conditional mode only)
        - "custom_{group_name}_found": bool for each custom ion group (if detect_custom=True)
        - "custom_{group_name}_max_intensity": max intensity for each group (if detect_custom=True)
        
    Raises
    ------
    ValueError
        If required columns are missing from DataFrame
        
    Notes
    -----
    - Uses caching for theoretical spectra to avoid recomputation
    - For large datasets, consider processing in batches
    
    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "sequence": ["PEPTIDE", "EXAMPLE"],
    ...     "precursor_charge": [2, 3],
    ...     "mz_array": [[100.0, 200.0], [150.0, 250.0]],
    ...     "intensity_array": [[1000, 2000], [500, 1500]]
    ... })
    >>> annotated = annotate_dataframe(
    ...     df,
    ...     ppm_tol=10.0,
    ...     ion_types=["b", "y"],
    ...     detect_custom=True
    ... )
    >>> "signal_mask" in annotated.columns
    True
    """
    # Validate required columns
    required_cols = {"sequence", "precursor_charge", "mz_array", "intensity_array"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    # Cache for theoretical spectra: (sequence, charge) -> (mz_array, annotations)
    theo_cache: Dict[Tuple[str, int], Tuple[np.ndarray, List[str]]] = {}

    # Output lists
    signal_masks: List[List[bool]] = []
    ppm_errors: List[List[float]] = []
    matched_annotations: List[List[Optional[str]]] = []
    matched_theo_mzs: List[List[float]] = []
    n_matched: List[int] = []
    frac_intensities: List[float] = []
    median_abs_ppms: List[float] = []
    mean_ppm_biases: List[float] = []
    
    # Conditional mode specific outputs
    feature_types: Optional[List[List[Optional[str]]]] = [] if use_conditional_annotation else None
    parent_annotations: Optional[List[List[Optional[str]]]] = [] if use_conditional_annotation else None
    n_base_list: Optional[List[int]] = [] if use_conditional_annotation else None
    n_losses_list: Optional[List[int]] = [] if use_conditional_annotation else None
    n_isotopes_list: Optional[List[int]] = [] if use_conditional_annotation else None
    
    # Custom ion detection tracking
    custom_results: Dict[str, List[Any]] = {}
    if detect_custom:
        if custom_ions is None:
            custom_ions = DEFAULT_CUSTOM_IONS
        for group_name in custom_ions:
            custom_results[f"{group_name}_found"] = []
            custom_results[f"{group_name}_n_matched"] = []
            custom_results[f"{group_name}_max_intensity"] = []
    
    # Build iterator
    data_iter = zip(
        df["sequence"].to_list(),
        df["precursor_charge"].to_list(),
        df["mz_array"].to_list(),
        df["intensity_array"].to_list(),
    )
    
    # Add progress bar if requested
    if progress:
        try:
            from tqdm.auto import tqdm
            data_iter = tqdm(
                data_iter,
                total=len(df),
                desc="Annotating spectra",
                unit="spectra",
            )
        except ImportError:
            warnings.warn(
                "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                stacklevel=2,
            )
    
    # Process each spectrum
    for seq, charge, mz_arr, int_arr in data_iter:
        # Convert to numpy
        mz_exp = np.asarray(mz_arr, dtype=np.float64)
        int_exp = np.asarray(int_arr, dtype=np.float64)
        
        # Choose annotation strategy
        if use_conditional_annotation:
            # Use two-pass conditional annotation
            try:
                match_result = match_with_conditional_features(
                    exp_mz=mz_exp,
                    exp_intensity=int_exp,
                    peptide=seq,
                    precursor_charge=charge,
                    ppm_tol=ppm_tol,
                    da_tol=da_tol,
                    ion_types=ion_types,
                    max_charge=max_charge,
                    add_losses=add_losses,
                    loss_types=loss_types,
                    add_isotopes=add_isotopes,
                    max_isotope=max_isotope,
                    isotope_intensity_threshold=isotope_intensity_threshold,
                    add_precursor=add_precursor,
                    use_closest=use_closest_match,
                    engine=engine,
                    fragmentation_type=fragmentation_type,
                )
            except Exception as e:
                # Handle invalid sequences gracefully
                warnings.warn(
                    f"Failed to annotate spectrum for '{seq}': {e}",
                    stacklevel=2,
                )
                # Use empty arrays for failed cases
                signal_masks.append([False] * len(mz_exp))
                ppm_errors.append([np.nan] * len(mz_exp))
                matched_annotations.append([None] * len(mz_exp))
                matched_theo_mzs.append([np.nan] * len(mz_exp))
                n_matched.append(0)
                frac_intensities.append(0.0)
                median_abs_ppms.append(np.nan)
                mean_ppm_biases.append(np.nan)
                if feature_types is not None:
                    feature_types.append([None] * len(mz_exp))
                if parent_annotations is not None:
                    parent_annotations.append([None] * len(mz_exp))
                if n_base_list is not None:
                    n_base_list.append(0)
                if n_losses_list is not None:
                    n_losses_list.append(0)
                if n_isotopes_list is not None:
                    n_isotopes_list.append(0)
                
                if detect_custom and custom_ions is not None:
                    for group_name in custom_ions:
                        custom_results[f"{group_name}_found"].append(False)
                        custom_results[f"{group_name}_n_matched"].append(0)
                        custom_results[f"{group_name}_max_intensity"].append(0.0)
                continue
            
            signal_masks.append(match_result["mask"].tolist())
            ppm_errors.append(match_result["ppm_error"].tolist())
            matched_annotations.append(match_result["matched_annotation"])
            matched_theo_mzs.append(match_result["matched_theo_mz"].tolist())
            n_matched.append(match_result["metrics"]["n_matched"])
            frac_intensities.append(match_result["metrics"]["frac_intensity"])
            median_abs_ppms.append(match_result["metrics"]["median_abs_ppm"])
            mean_ppm_biases.append(match_result["metrics"]["mean_ppm_bias"])
            if feature_types is not None:
                feature_types.append(match_result["feature_type"])
            if parent_annotations is not None:
                parent_annotations.append(match_result["parent_annotation"])
            if n_base_list is not None:
                n_base_list.append(match_result["metrics"]["n_base"])
            if n_losses_list is not None:
                n_losses_list.append(match_result["metrics"]["n_losses"])
            if n_isotopes_list is not None:
                n_isotopes_list.append(match_result["metrics"]["n_isotopes"])
            
        else:
            # Use traditional approach with all features generated upfront
            # Get or generate theoretical spectrum
            cache_key = (seq, charge)
            if cache_key in theo_cache:
                theo_mz, theo_annotations = theo_cache[cache_key]
            else:
                try:
                    theo_mz, theo_annotations = generate_theoretical_spectrum(
                        peptide=seq,
                        precursor_charge=charge,
                        ion_types=ion_types,
                        max_charge=max_charge,
                        add_losses=add_losses,
                        add_isotopes=add_isotopes,
                        max_isotope=max_isotope,
                        isotope_model=isotope_model,
                        max_isotope_probability=max_isotope_probability,
                        add_precursor=add_precursor,
                        add_custom_ions=False,  # Don't include custom ions in matching
                        engine=engine,
                        fragmentation_type=fragmentation_type,
                    )
                    theo_cache[cache_key] = (theo_mz, theo_annotations)
                except Exception as e:
                    # Handle invalid sequences gracefully
                    warnings.warn(
                        f"Failed to generate theoretical spectrum for '{seq}': {e}",
                        stacklevel=2,
                    )
                    # Use empty arrays for failed cases
                    signal_masks.append([False] * len(mz_exp))
                    ppm_errors.append([np.nan] * len(mz_exp))
                    matched_annotations.append([None] * len(mz_exp))
                    matched_theo_mzs.append([np.nan] * len(mz_exp))
                    n_matched.append(0)
                    frac_intensities.append(0.0)
                    median_abs_ppms.append(np.nan)
                    mean_ppm_biases.append(np.nan)
                    
                    if detect_custom and custom_ions is not None:
                        for group_name in custom_ions:
                            custom_results[f"{group_name}_found"].append(False)
                            custom_results[f"{group_name}_n_matched"].append(0)
                            custom_results[f"{group_name}_max_intensity"].append(0.0)
                    continue
            
            # Match theoretical to experimental
            match_result = match_theoretical_to_experimental(
                mz_exp,
                int_exp,
                theo_mz,
                ppm_tol=ppm_tol,
                da_tol=da_tol,
                use_closest=use_closest_match,
                theo_annotations=theo_annotations,
            )
            
            signal_masks.append(match_result["mask"].tolist())
            ppm_errors.append(match_result["ppm_error"].tolist())
            matched_annotations.append(match_result["matched_annotation"])
            matched_theo_mzs.append(match_result["matched_theo_mz"].tolist())
            n_matched.append(match_result["metrics"]["n_matched"])
            frac_intensities.append(match_result["metrics"]["frac_intensity"])
            median_abs_ppms.append(match_result["metrics"]["median_abs_ppm"])
            mean_ppm_biases.append(match_result["metrics"]["mean_ppm_bias"])
        
        # Detect custom ions if requested
        if detect_custom:
            custom_result = detect_custom_ions(
                mz_exp,
                int_exp,
                custom_ions=custom_ions,
                ppm_tol=ppm_tol,
                return_details=False,
            )
            if custom_ions is not None:
                for group_name in custom_ions:
                    custom_results[f"{group_name}_found"].append(
                        custom_result[group_name]["found"]
                    )
                    custom_results[f"{group_name}_n_matched"].append(
                        custom_result[group_name]["n_matched"]
                    )
                    custom_results[f"{group_name}_max_intensity"].append(
                        custom_result[group_name]["max_intensity"]
                    )
    
    # Build result columns
    result_cols = [
        pl.Series("signal_mask", signal_masks, dtype=pl.List(pl.Boolean)),
        pl.Series("ppm_error", ppm_errors, dtype=pl.List(pl.Float64)),
        pl.Series("matched_annotation", matched_annotations, dtype=pl.List(pl.Utf8)),
        pl.Series("matched_theo_mz", matched_theo_mzs, dtype=pl.List(pl.Float64)),
        pl.Series("n_matched", n_matched, dtype=pl.UInt32),
        pl.Series("frac_intensity", frac_intensities, dtype=pl.Float64),
        pl.Series("median_abs_ppm", median_abs_ppms, dtype=pl.Float64),
        pl.Series("mean_ppm_bias", mean_ppm_biases, dtype=pl.Float64),
    ]
    
    # Add conditional mode specific columns
    if use_conditional_annotation:
        result_cols.extend([
            pl.Series("feature_type", feature_types, dtype=pl.List(pl.Utf8)),
            pl.Series("parent_annotation", parent_annotations, dtype=pl.List(pl.Utf8)),
            pl.Series("n_base", n_base_list, dtype=pl.UInt32),
            pl.Series("n_losses", n_losses_list, dtype=pl.UInt32),
            pl.Series("n_isotopes", n_isotopes_list, dtype=pl.UInt32),
        ])
    
    # Add custom ion detection columns
    if detect_custom:
        for col_name, values in custom_results.items():
            if col_name.endswith("_found"):
                result_cols.append(pl.Series(f"custom_{col_name}", values, dtype=pl.Boolean))
            elif col_name.endswith("_n_matched"):
                result_cols.append(pl.Series(f"custom_{col_name}", values, dtype=pl.UInt32))
            else:  # max_intensity
                result_cols.append(pl.Series(f"custom_{col_name}", values, dtype=pl.Float64))
    
    return df.with_columns(result_cols)
