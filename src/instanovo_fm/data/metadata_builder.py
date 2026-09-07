"""Metadata builder for foundation model meta token.

Constructs the metadata dictionary from batch data for consumption by MetaTokenEmbed.
Handles categorical encoding, binning, and missing value fallbacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


# ===== Categorical Vocabularies =====

FRAG_TYPES = ["HCD", "HCID", "CID", "ETD"]

INSTRUMENTS = [
    'Orbitrap Eclipse', 'Q Exactive HFX', 'Orbitrap Elite', 'Orbitrap Velos',
    'Orbitrap Q Exactive', 'Orbitrap Fusion', 'Q Exactive HF', 'Q Exactive Plus',
    'timsTOF Pro 2', 'Orbitrap Fusion Lumos', 'Agilent 6400 Series Triple Quadrupole LC/MS',
    'Orbitrap Exploris 480', 'Q Exactive', 'timsTOF SCP', 'Orbitrap Exploris 240',
    'Orbitrap Astral', 'timsTOF', 'TripleTOF 6600', 'Synapt MS', 'Orbitrap Ascend'
]

DETECTORS = [
    'Orbitrap', 'Iontrap', 'Orbitrap|IonTrap', 'TOF',
    'Triple Quadrupole', 'Triple quadrupole', 'Astral'
]

ENZYMES = [
    'trypsin', 'unspecific', 'aspn', 'lysn',
    'chymotrypsin', 'lysc', 'gluc'
]

ACQUISITIONS = ['DIA', 'DDA']

QUANT_METHODS = ['precursor', 'TMT', 'iTRAQ']


# ===== Binning Configurations =====

# Precursor charge: bin_size=1, range 1-7+
# Bins: [0: charge 1, 1: charge 2, 2: charge 3, 3: charge 4, 4: charge 5, 5: charge 6, 6: charge 7+]
CHARGE_BIN_EDGES = [1, 2, 3, 4, 5, 6, 7, 8]  # Right edges for bucketize
CHARGE_N_BINS = 7  # 7 bins: charge 1, 2, 3, 4, 5, 6, 7+

# Collision energy: continuous (no longer binned)
# Observed range in data: 15-40 NCE
# We normalize to this range for better resolution in [0, 1]
CE_MIN = 15.0   # Minimum NCE (observed in data)
CE_MAX = 40.0   # Maximum NCE (observed in data)


# Pre-built vocabulary lookup dictionaries (loaded once at module import)
_VOCAB_LOOKUPS = {
    'FRAG_TYPES': {v.lower(): i for i, v in enumerate(FRAG_TYPES)},
    'INSTRUMENTS': {v.lower(): i for i, v in enumerate(INSTRUMENTS)},
    'DETECTORS': {v.lower(): i for i, v in enumerate(DETECTORS)},
    'ENZYMES': {v.lower(): i for i, v in enumerate(ENZYMES)},
    'ACQUISITIONS': {v.lower(): i for i, v in enumerate(ACQUISITIONS)},
    'QUANT_METHODS': {v.lower(): i for i, v in enumerate(QUANT_METHODS)},
}


def _encode_categorical(
    values: List[Any],
    vocab_lookup: Dict[str, int],
    default_id: int = 0,
) -> torch.Tensor:
    """Encode categorical values to indices using pre-built lookup dictionary.
    
    Args:
        values: List of raw categorical values.
        vocab_lookup: Pre-built {value: id} lookup dictionary.
        default_id: Default ID for unknown values (default: 0).
    
    Returns:
        Tensor of indices, shape (B,).
    """
    indices = [
        vocab_lookup.get(str(val).lower(), default_id) if val is not None else default_id
        for val in values
    ]
    return torch.tensor(indices, dtype=torch.long)


# Pre-computed binning tensors (loaded once at module import)
_CHARGE_BIN_EDGES_TENSOR = torch.tensor(CHARGE_BIN_EDGES, dtype=torch.float32)
_CHARGE_N_BINS = len(CHARGE_BIN_EDGES) - 1


def _bin_values(
    values: List[float],
    edges_tensor: torch.Tensor,
    n_bins: int,
    default_id: int = 0,
) -> torch.Tensor:
    """Bin continuous values into discrete categories using vectorized operations.
    
    Args:
        values: List of continuous values.
        edges_tensor: Pre-computed bin edges tensor.
        n_bins: Number of bins (len(edges) - 1).
        default_id: Default bin ID for None/NaN values (default: 0).
    
    Returns:
        Tensor of bin indices, shape (B,).
    """
    # Convert values to tensor, replacing None with NaN for vectorized handling
    values_clean = [float(v) if v is not None else float('nan') for v in values]
    values_tensor = torch.tensor(values_clean, dtype=torch.float32)
    
    # Identify missing values (NaN or inf)
    missing_mask = ~torch.isfinite(values_tensor)
    
    # Replace NaN/inf with valid value for bucketize (will be overridden anyway)
    values_tensor = torch.where(missing_mask, edges_tensor[0], values_tensor)
    
    # Bucketize: returns bin index in [0, len(edges)]
    bin_ids = torch.bucketize(values_tensor, edges_tensor, right=False)
    
    # Clamp to valid range [0, n_bins - 1] so indices stay within embedding vocab
    bin_ids = bin_ids.clamp(0, n_bins - 1)
    
    # Override missing values with default_id
    bin_ids[missing_mask] = default_id
    
    return bin_ids


# Pre-computed CE normalization constant (loaded once at module import)
_CE_SCALE = 1.0 / (CE_MAX - CE_MIN)


def build_metadata_dict(
    batch: List[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build metadata dictionary for MetaTokenEmbed from batch data.
    
    Extracts and encodes metadata fields from batch into tensors for the meta token.
    Uses pre-computed lookups and vectorized operations for optimal performance.
    
    Args:
        batch: List of sample dictionaries (each with metadata fields).
        device: Target device for tensors (default: CPU).
    
    Returns:
        Dictionary with encoded metadata tensors:
            Continuous:
            - precursor_mass: (B,) [neutral mass in Da]
            - collision_energy: (B,) [NCE, normalized to [0,1]]
            
            Categorical:
            - frag_type: (B,)
            - search_instrument: (B,)
            - search_acquisition: (B,)
            - search_detector: (B,)
            - search_enzyme: (B,)
            - search_quant: (B,)
            - precursor_charge: (B,) [binned]
    """
    if not batch:
        return {}
    
    device = device or torch.device("cpu")
    meta = {}
    
    # ===== Continuous Features =====
    
    # Precursor mass (neutral mass, more informative than m/z)
    # Prefer pre-calculated precursor_mass from batch for consistency
    if "precursor_mass" in batch[0]:
        # Direct precursor mass available - fast path
        # Handle None values inline for speed
        precursor_mass_list = [
            float(sample.get("precursor_mass", 0.0)) if sample.get("precursor_mass") is not None else 0.0
            for sample in batch
        ]
        meta["precursor_mass"] = torch.tensor(precursor_mass_list, dtype=torch.float32, device=device)
    elif "precursor_mz" in batch[0] and "precursor_charge" in batch[0]:
        # Fallback: Calculate from m/z and charge
        from instanovo.constants import PROTON_MASS_AMU
        
        # Fast path: list comprehensions with inline validation
        # Replace None/invalid with safe defaults to avoid FPE
        mz_list = [
            float(sample.get("precursor_mz", 0.0)) if (
                (mz := sample.get("precursor_mz")) is not None and 
                isinstance(mz, (int, float))
            ) else 0.0
            for sample in batch
        ]
        charge_list = [
            float(sample.get("precursor_charge", 1)) if (
                (c := sample.get("precursor_charge")) is not None and 
                isinstance(c, (int, float)) and 
                c > 0
            ) else 1.0
            for sample in batch
        ]
        
        mz_tensor = torch.tensor(mz_list, dtype=torch.float32, device=device)
        charge_tensor = torch.tensor(charge_list, dtype=torch.float32, device=device)
        
        # Vectorized: mass = mz * charge - charge * PROTON_MASS_AMU
        meta["precursor_mass"] = mz_tensor * charge_tensor - charge_tensor * PROTON_MASS_AMU
    
    # Collision energy (continuous, normalized to [0, 1])
    if "collision_energy" in batch[0]:
        # Fast path: use list comprehension with inline None handling
        ce_list = [
            float(sample.get("collision_energy", CE_MIN)) if sample.get("collision_energy") is not None else CE_MIN
            for sample in batch
        ]
        
        ce_tensor = torch.tensor(ce_list, dtype=torch.float32, device=device)
        
        # Vectorized normalization and clamping
        # (x - CE_MIN) * _CE_SCALE is faster than (x - CE_MIN) / (CE_MAX - CE_MIN)
        meta["collision_energy"] = ((ce_tensor - CE_MIN) * _CE_SCALE).clamp_(0.0, 1.0)
    
    # ===== Categorical Features =====
    
    # Fragmentation type
    if "frag_type" in batch[0]:
        frag_list = [sample.get("frag_type") for sample in batch]
        meta["frag_type"] = _encode_categorical(frag_list, _VOCAB_LOOKUPS['FRAG_TYPES']).to(device)
    
    # Instrument model
    if "search_instrument" in batch[0]:
        instrument_list = [sample.get("search_instrument") for sample in batch]
        meta["search_instrument"] = _encode_categorical(instrument_list, _VOCAB_LOOKUPS['INSTRUMENTS']).to(device)
    
    # Acquisition mode
    if "search_acquisition" in batch[0]:
        acq_list = [sample.get("search_acquisition") for sample in batch]
        meta["search_acquisition"] = _encode_categorical(acq_list, _VOCAB_LOOKUPS['ACQUISITIONS']).to(device)
    
    # Detector type
    if "search_detector" in batch[0]:
        detector_list = [sample.get("search_detector") for sample in batch]
        meta["search_detector"] = _encode_categorical(detector_list, _VOCAB_LOOKUPS['DETECTORS']).to(device)
    
    # Enzyme
    if "search_enzyme" in batch[0]:
        enzyme_list = [sample.get("search_enzyme") for sample in batch]
        meta["search_enzyme"] = _encode_categorical(enzyme_list, _VOCAB_LOOKUPS['ENZYMES']).to(device)
    
    # Quantification method
    if "search_quant" in batch[0]:
        quant_list = [sample.get("search_quant") for sample in batch]
        meta["search_quant"] = _encode_categorical(quant_list, _VOCAB_LOOKUPS['QUANT_METHODS']).to(device)
    
    # ===== Binned Features =====
    
    # Precursor charge (binned: 1-7+)
    if "precursor_charge" in batch[0]:
        # Fast path: list comprehension with inline validation
        # Replace None/invalid with 2 (most common charge state)
        charge_list = [
            float(sample.get("precursor_charge", 2)) if (
                (c := sample.get("precursor_charge")) is not None and 
                isinstance(c, (int, float)) and 
                c > 0
            ) else 2.0
            for sample in batch
        ]
        meta["precursor_charge"] = _bin_values(charge_list, _CHARGE_BIN_EDGES_TENSOR, _CHARGE_N_BINS).to(device)
    
    return meta


def get_vocabulary_sizes() -> Dict[str, int]:
    """Get vocabulary sizes for all categorical features.
    
    Returns:
        Dictionary mapping feature names to vocabulary sizes.
    """
    return {
        "n_frag": len(FRAG_TYPES),
        "n_instrument": len(INSTRUMENTS),
        "n_acquisition": len(ACQUISITIONS),
        "n_detector": len(DETECTORS),
        "n_enzyme": len(ENZYMES),
        "n_quant": len(QUANT_METHODS),
        "n_charge": CHARGE_N_BINS,
    }

