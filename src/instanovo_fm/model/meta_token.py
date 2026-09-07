"""Metadata token embedding for InstaNovo Foundation Model.

Refactored implementation for encoding experimental metadata into separate tokens,
one per metadata field. Each field gets its own token that conditions the model
on specific acquisition parameters and instrument settings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn


class FourierScalarEncoder(nn.Module):
    """Sinusoidal Fourier encoding for continuous scalar values.

    Uses log-spaced frequencies for better multi-scale representation.

    Args:
        n_freq: Number of frequencies (default: 16).
        logspace: Use log-spaced frequencies (default: True).
    """

    def __init__(self, n_freq: int = 16, logspace: bool = True) -> None:
        """Initialise the input."""
        super().__init__()
        if logspace:
            freqs = torch.logspace(0, math.log10(n_freq), n_freq)
        else:
            freqs = torch.arange(n_freq).float()
        self.register_buffer("freqs", freqs * 2 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Scalar values in [0, 1] range, shape (B,).

        Returns: Fourier features, shape (B, 2*n_freq).
        """
        freqs: torch.Tensor = self.freqs.to(dtype=x.dtype, device=x.device)  # type: ignore[assignment]
        x_expanded = x.unsqueeze(-1) * freqs  # (B, n_freq)
        return torch.cat((x_expanded.sin(), x_expanded.cos()), dim=-1)  # (B, 2*n_freq)


class MetaTokenEmbed(nn.Module):
    """Metadata token embedding for proteomic spectra.

    Refactored to create separate tokens for each metadata field. Each field
    gets its own learnable token that conditions the model on specific aspects
    of the acquisition context.

    Features:
        - Continuous features: Fourier-encoded with configurable normalization
        - Categorical features: Learned embeddings with automatic vocab sizing
        - Flexible field inclusion via configuration
        - Robust handling of missing values
        - Each field gets its own token (not concatenated into one)

    Args:
        proj_dim: Output token dimension (matches model dimension).
        n_freq: Number of Fourier frequencies for continuous features (default: 16).

        Categorical vocabulary sizes:
        - n_frag: Number of fragmentation types (default: 4).
        - n_instrument: Number of instrument models (default: 20).
        - n_acquisition: Number of acquisition modes (default: 2).
        - n_detector: Number of detector types (default: 7).
        - n_enzyme: Number of enzymes (default: 7).
        - n_quant: Number of quantification methods (default: 3).
        - n_charge: Number of charge bins (default: 7, for charges 1-7+).

        Feature normalization bounds (for continuous features):
        - precursor_mass_max: Maximum precursor mass in Da (default: 6000.0, based on observed data).

        Field inclusion flags:
        - include_frag_type: Include fragmentation type (default: True).
        - include_instrument: Include instrument model (default: True).
        - include_acquisition: Include acquisition mode (DDA/DIA) (default: True).
        - include_detector: Include detector type (default: True).
        - include_enzyme: Include enzyme (default: True).
        - include_quant: Include quantification method (default: True).
        - include_precursor_charge: Include precursor charge (default: True).
        - include_precursor_mass: Include precursor mass (default: True).
        - include_collision_energy: Include collision energy (default: True).
    """

    def __init__(
        self,
        proj_dim: int,
        n_freq: int = 16,
        # Categorical vocabulary sizes
        n_frag: int = 4,
        n_instrument: int = 20,
        n_acquisition: int = 2,
        n_detector: int = 7,
        n_enzyme: int = 7,
        n_quant: int = 3,
        n_charge: int = 7,  # Charges 1-7+ (7 bins)
        # Continuous feature bounds
        precursor_mass_max: float = 6000.0,  # Observed max in training data
        # Field inclusion
        include_frag_type: bool = True,
        include_instrument: bool = True,
        include_acquisition: bool = True,
        include_detector: bool = True,
        include_enzyme: bool = True,
        include_quant: bool = True,
        include_precursor_charge: bool = True,
        include_precursor_mass: bool = True,
        include_collision_energy: bool = True,
    ) -> None:
        """Initialise the input."""
        super().__init__()

        # Store configuration
        self.proj_dim = proj_dim
        self.n_freq = n_freq
        self.precursor_mass_max = precursor_mass_max

        # Fourier encoder for continuous features (shared)
        self._fourier_encoder = FourierScalarEncoder(n_freq=n_freq, logspace=True)

        # Feature inclusion flags
        self.include_frag_type = include_frag_type
        self.include_instrument = include_instrument
        self.include_acquisition = include_acquisition
        self.include_detector = include_detector
        self.include_enzyme = include_enzyme
        self.include_quant = include_quant
        self.include_precursor_charge = include_precursor_charge
        self.include_precursor_mass = include_precursor_mass
        self.include_collision_energy = include_collision_energy

        # Build individual token embeddings for each field
        self.token_embeddings = nn.ModuleDict()

        # Categorical field embeddings
        if include_frag_type:
            self.token_embeddings["frag_type"] = nn.Sequential(
                nn.Embedding(n_frag, 64),
                nn.Linear(64, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_instrument:
            self.token_embeddings["instrument"] = nn.Sequential(
                nn.Embedding(n_instrument, 128),
                nn.Linear(128, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_acquisition:
            self.token_embeddings["acquisition"] = nn.Sequential(
                nn.Embedding(n_acquisition, 32),
                nn.Linear(32, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_detector:
            self.token_embeddings["detector"] = nn.Sequential(
                nn.Embedding(n_detector, 64),
                nn.Linear(64, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_enzyme:
            self.token_embeddings["enzyme"] = nn.Sequential(
                nn.Embedding(n_enzyme, 64),
                nn.Linear(64, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_quant:
            self.token_embeddings["quant"] = nn.Sequential(
                nn.Embedding(n_quant, 32),
                nn.Linear(32, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_precursor_charge:
            self.token_embeddings["precursor_charge"] = nn.Sequential(
                nn.Embedding(n_charge, 64),
                nn.Linear(64, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        # Continuous field embeddings (Fourier-encoded)
        fourier_dim = 2 * n_freq  # sin + cos components

        if include_precursor_mass:
            self.token_embeddings["precursor_mass"] = nn.Sequential(
                nn.Linear(fourier_dim, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        if include_collision_energy:
            self.token_embeddings["collision_energy"] = nn.Sequential(
                nn.Linear(fourier_dim, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )

        # Store enabled fields in order (for consistent token ordering)
        self.enabled_fields: List[str] = []
        if include_frag_type:
            self.enabled_fields.append("frag_type")
        if include_instrument:
            self.enabled_fields.append("instrument")
        if include_acquisition:
            self.enabled_fields.append("acquisition")
        if include_detector:
            self.enabled_fields.append("detector")
        if include_enzyme:
            self.enabled_fields.append("enzyme")
        if include_quant:
            self.enabled_fields.append("quant")
        if include_precursor_charge:
            self.enabled_fields.append("precursor_charge")
        if include_precursor_mass:
            self.enabled_fields.append("precursor_mass")
        if include_collision_energy:
            self.enabled_fields.append("collision_energy")

    def _normalize_continuous(self, x: torch.Tensor, max_val: float) -> torch.Tensor:
        """Normalize continuous feature to [0, 1] range.

        Args:
            x: Raw feature values.
            max_val: Maximum value for normalization.

        Returns:
            Normalized values in [0, 1] range.
        """
        # Handle NaN/Inf
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        # Normalize and clamp
        normalized = (x / max_val).clamp(0.0, 1.0)
        return normalized

    def _get_categorical_id(
        self,
        meta: Dict[str, torch.Tensor],
        meta_key: str,
        vocab_size: int,
        default_id: int = 0,
    ) -> torch.Tensor:
        """Extract categorical ID from metadata with robust error handling.

        Args:
            meta: Metadata dictionary.
            meta_key: Metadata key to extract.
            vocab_size: Vocabulary size for clamping.
            default_id: Default ID if key is missing (default: 0).

        Returns:
            Categorical IDs as long tensor, shape (B,).
        """
        if meta_key not in meta:
            # Missing feature - use default
            batch_size = next(iter(meta.values())).shape[0]
            device = next(iter(meta.values())).device
            return torch.full((batch_size,), default_id, dtype=torch.long, device=device)

        ids = meta[meta_key]
        # Convert to long dtype and clamp to valid range
        ids = ids.to(dtype=torch.long).clamp(0, vocab_size - 1)
        return ids

    def forward(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode metadata into separate token embeddings, one per field.

        Args:
            meta: Metadata dictionary with fields:
                Continuous:
                - precursor_mass: Neutral precursor mass in Da (B,)
                - collision_energy: Normalized collision energy [0,1] (B,)

                Categorical (as indices):
                - frag_type: Fragmentation type IDs (B,)
                - search_instrument: Instrument model IDs (B,)
                - search_acquisition: Acquisition mode IDs (B,)
                - search_detector: Detector type IDs (B,)
                - search_enzyme: Enzyme IDs (B,)
                - search_quant: Quantification method IDs (B,)
                - precursor_charge: Charge state IDs (B,) [binned 2-6]

        Returns:
            Meta token embeddings, shape (B, n_tokens, proj_dim) where n_tokens
            is the number of enabled metadata fields.
        """
        if not meta:
            raise ValueError("Empty metadata dictionary provided")

        device = next(iter(meta.values())).device
        batch_size = next(iter(meta.values())).shape[0]

        tokens = []

        # Map metadata keys to field names
        meta_key_mapping = {
            "frag_type": "frag_type",
            "instrument": "search_instrument",
            "acquisition": "search_acquisition",
            "detector": "search_detector",
            "enzyme": "search_enzyme",
            "quant": "search_quant",
            "precursor_charge": "precursor_charge",
            "precursor_mass": "precursor_mass",
            "collision_energy": "collision_energy",
        }

        # Process each enabled field in order
        for field_name in self.enabled_fields:
            if field_name not in self.token_embeddings:
                continue

            if field_name in ["precursor_mass", "collision_energy"]:
                # Continuous field: Fourier encode then project
                meta_key = meta_key_mapping[field_name]

                if field_name == "precursor_mass":
                    if meta_key in meta:
                        normalized = self._normalize_continuous(meta[meta_key], self.precursor_mass_max)
                    else:
                        normalized = torch.zeros(batch_size, device=device)
                    fourier_features = self._fourier_encoder(normalized)
                    token = self.token_embeddings[field_name](fourier_features)
                else:  # collision_energy
                    if meta_key in meta:
                        # Already normalized in metadata_builder
                        ce_normalized = meta[meta_key]
                    else:
                        ce_normalized = torch.zeros(batch_size, device=device)
                    fourier_features = self._fourier_encoder(ce_normalized)
                    token = self.token_embeddings[field_name](fourier_features)

                tokens.append(token)
            else:
                # Categorical field: Embed then project
                meta_key = meta_key_mapping[field_name]

                # Get vocabulary size from the embedding layer
                embedding_layer = self.token_embeddings[field_name][0]  # First layer is Embedding
                vocab_size = embedding_layer.num_embeddings  # type: ignore[assignment]

                ids = self._get_categorical_id(meta, meta_key, vocab_size, default_id=0)
                token = self.token_embeddings[field_name](ids)
                tokens.append(token)

        if not tokens:
            # No tokens enabled - return empty tensor
            return torch.zeros(batch_size, 0, self.proj_dim, device=device)

        # Stack tokens: (B, n_tokens, proj_dim)
        return torch.stack(tokens, dim=1)

    def get_feature_info(self) -> Dict[str, Any]:
        """Return configuration summary for debugging/inspection.

        Returns:
            Dictionary with feature configuration and dimensions.
        """
        continuous_features = []
        categorical_features = []

        for field_name in self.enabled_fields:
            if field_name in ["precursor_mass", "collision_energy"]:
                continuous_features.append(field_name)
            else:
                categorical_features.append(field_name)

        return {
            "proj_dim": self.proj_dim,
            "n_freq": self.n_freq,
            "enabled_fields": self.enabled_fields,
            "enabled_continuous": continuous_features,
            "enabled_categorical": categorical_features,
            "n_tokens": len(self.enabled_fields),
        }
