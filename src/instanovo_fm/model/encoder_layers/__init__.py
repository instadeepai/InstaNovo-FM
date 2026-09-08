"""Encoder layer implementations for different attention mechanisms.

This package contains specialized encoder layers for:
- Unified: Modern unified encoder with configurable attention mechanisms
- Support for various positional encodings (sinusoidal, RoPE, none)
- Support for various relative biases (ALiBi, RPE, Graph, none)
- Support for different attention backends (Flash Attention, BiasAware)
"""

from instanovo_fm.model.positional.alibi import ALiBi

from .factories import (
    create_attention_mechanism,
    create_positional_encoding,
    create_relative_bias,
    create_unified_encoder_stack,
    parse_architecture_config,
    validate_architecture_config,
)
from .unified_encoder import UnifiedEncoderLayer, UnifiedTransformerEncoder

__all__ = [
    # Relative bias components (used by unified encoder)
    "ALiBi",
    # New unified encoder
    "UnifiedEncoderLayer",
    "UnifiedTransformerEncoder",
    # Factory functions
    "create_positional_encoding",
    "create_relative_bias",
    "create_attention_mechanism",
    "parse_architecture_config",
    "validate_architecture_config",
    "create_unified_encoder_stack",
]
