"""Factory functions for creating encoder components.

This module provides factory functions for creating:
- Positional encodings (sinusoidal, RoPE, none)
- Relative biases (ALiBi, RPE, PA, none)
- Attention mechanisms (Flash, BiasAware)
- Encoder layers and stacks

This centralizes the component creation logic and makes the architecture more modular.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from instanovo.constants import MAX_SEQUENCE_LENGTH
from instanovo_fm.model.attention.bias_aware import BiasAwareMHA
from instanovo_fm.model.attention.flash import FlashMHA
from instanovo_fm.model.pairwise_bias import PairwiseAttentionBias
from instanovo_fm.model.positional.rotary import SimpleRotaryEmbedding
from instanovo_fm.model.positional.sinusoidal import PositionalEncoding

logger = logging.getLogger(__name__)


def create_positional_encoding(pos_type: str, d_model: int, n_heads: int, config: Dict[str, Any]) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """Factory function for creating positional encodings.

    Args:
        pos_type: Type of positional encoding ('sinusoidal', 'rope', 'none')
        d_model: Model dimension
        n_heads: Number of attention heads
        config: Configuration dictionary for the positional encoding

    Returns:
        Tuple of (pos_embed, rotary_emb) where one will be None
    """
    if pos_type == "sinusoidal":
        # Add buffer for special tokens (latent + meta)
        # Default max_len needs to accommodate n_peaks + special tokens
        max_len = config.get("max_len", MAX_SEQUENCE_LENGTH + 10)
        dropout = config.get("dropout", 0.1)
        pos_embed = PositionalEncoding(d_model, dropout, max_len)
        return pos_embed, None

    elif pos_type == "rope":
        head_dim = d_model // n_heads
        rotary_pct = config.get("rotary_pct", 1.0)
        base = config.get("base", 10000.0)
        rotary_emb = SimpleRotaryEmbedding(head_dim, rotary_pct=rotary_pct, base=base)
        return None, rotary_emb

    elif pos_type == "none":
        return None, None

    else:
        raise ValueError(f"Unknown positional encoding type: {pos_type}")


def create_relative_bias(bias_type: str, n_heads: int, d_model: int, config: Dict[str, Any]) -> Optional[nn.Module]:
    """Factory function for creating relative bias modules.

    Args:
        bias_type: Type of relative bias ('alibi', 'rpe', 'pa', 'none')
        n_heads: Number of attention heads
        d_model: Model dimension
        config: Configuration dictionary for the relative bias

    Returns:
        Relative bias module or None
    """
    if bias_type == "alibi":
        from instanovo_fm.model.positional.alibi import ALiBi

        max_seq_len = config.get("max_seq_len", 200)
        return ALiBi(n_heads, max_seq_len)

    elif bias_type == "rpe":
        from instanovo_fm.model.positional.relative import RelativePositionalEncoding

        max_relative_position = config.get("max_relative_position", 32)
        return RelativePositionalEncoding(max_relative_position, d_model)

    elif bias_type == "pa":
        return PairwiseAttentionBias(
            num_freqs=config.get("pw_num_freqs", 16),
            hidden_dim=config.get("pw_hidden_dim", 16),
            lambda_min=config.get("lambda_min", 0.001),
            lambda_max=config.get("lambda_max", 10000.0),
        )

    elif bias_type == "none":
        return None

    else:
        raise ValueError(f"Unknown relative bias type: {bias_type}")


def create_attention_mechanism(
    bias_type: str, d_model: int, n_heads: int, dropout: float, rotary_emb: Optional[nn.Module], attn_backend: str
) -> Optional[nn.Module]:
    """Factory function for creating attention mechanisms.

    Args:
        bias_type: Type of relative bias (affects attention choice)
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        rotary_emb: Rotary embedding if using RoPE
        attn_backend: Attention backend ('flash' or 'math')

    Returns:
        Attention module or None (for standard PyTorch attention)
    """
    # FlashMHA uses F.scaled_dot_product_attention which dispatches to:
    #   - flash kernel (no bias, fastest)
    #   - memory-efficient kernel (with bias via float attn_mask)
    # BiasAwareMHA uses manual matmul→softmax (slower, but always works)
    if attn_backend == "flash":
        attention = FlashMHA(embed_dim=d_model, num_heads=n_heads, dropout=dropout, rotary_emb=rotary_emb)
        return attention

    # Explicit math backend: manual attention for full control
    attention = BiasAwareMHA(embed_dim=d_model, num_heads=n_heads, dropout=dropout, rotary_emb=rotary_emb)
    return attention


def parse_architecture_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse architecture configuration from config dictionary.

    Args:
        cfg: Configuration dictionary

    Returns:
        Parsed architecture configuration

    Raises:
        ValueError: If configuration does not use the modular architecture format
    """
    if "architecture" not in cfg:
        raise ValueError(
            "Configuration must use the modular architecture format. "
            "Add an 'architecture' section with 'positional_encoding', 'relative_bias', and 'attention' subsections."
        )

    arch_cfg = cfg["architecture"]
    pos_cfg = arch_cfg.get("positional_encoding", {"type": "sinusoidal"})
    bias_cfg = arch_cfg.get("relative_bias", {"type": "none"})
    attn_cfg = arch_cfg.get("attention", {"backend": "math"})

    # Extract bias config - convert to dict to avoid OmegaConf struct mode issues
    from omegaconf import OmegaConf

    raw_bias_config = bias_cfg.get("config", {})

    # Convert OmegaConf to regular dict if needed
    if hasattr(raw_bias_config, "_metadata"):  # Check if it's an OmegaConf object
        bias_config = OmegaConf.to_container(raw_bias_config, resolve=True)
    else:
        bias_config = dict(raw_bias_config) if raw_bias_config else {}

    # PA parameters with sensible defaults
    bias_config.setdefault("pw_num_freqs", 16)
    bias_config.setdefault("pw_hidden_dim", 16)
    bias_config.setdefault("per_layer_pw", True)
    bias_config.setdefault("lambda_min", 0.001)
    bias_config.setdefault("lambda_max", 10000.0)

    # Convert pos_config and attn_config to dicts as well for consistency
    raw_pos_config = pos_cfg.get("config", {})
    if hasattr(raw_pos_config, "_metadata"):
        pos_config = OmegaConf.to_container(raw_pos_config, resolve=True)
    else:
        pos_config = dict(raw_pos_config) if raw_pos_config else {}

    raw_attn_config = attn_cfg.get("config", {})
    if hasattr(raw_attn_config, "_metadata"):
        attn_config = OmegaConf.to_container(raw_attn_config, resolve=True)
    else:
        attn_config = dict(raw_attn_config) if raw_attn_config else {}

    return {
        "pos_type": pos_cfg.get("type", "sinusoidal"),
        "pos_config": pos_config,
        "bias_type": bias_cfg.get("type", "none"),
        "bias_config": bias_config,
        "attn_backend": attn_cfg.get("backend", "math"),
        "attn_config": attn_config,
        "is_modular": True,
    }


def validate_architecture_config(arch_config: Dict[str, Any]) -> None:
    """Validate architecture configuration.

    Args:
        arch_config: Architecture configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    pos_type = arch_config["pos_type"]
    bias_type = arch_config["bias_type"]
    attn_backend = arch_config["attn_backend"]

    # Validate positional encoding
    valid_pos_types = {"sinusoidal", "rope", "none"}
    if pos_type not in valid_pos_types:
        raise ValueError(f"Invalid positional encoding type: {pos_type}. Must be one of {valid_pos_types}")

    # Validate relative bias
    valid_bias_types = {"alibi", "alibi_pa", "rpe", "pa", "none"}
    if bias_type not in valid_bias_types:
        raise ValueError(f"Invalid relative bias type: {bias_type}. Must be one of {valid_bias_types}")

    # Validate attention backend
    valid_backends = {"flash", "math"}
    if attn_backend not in valid_backends:
        raise ValueError(f"Invalid attention backend: {attn_backend}. Must be one of {valid_backends}")

    if bias_type == "pa" and pos_type == "sinusoidal":
        logger.warning("PA bias with sinusoidal positional encoding may not be optimal. Consider using RoPE.")


def create_unified_encoder_stack(
    cfg: Dict[str, Any], d_model: int, n_heads: int, dim_feedforward: int, dropout: float, n_layers: int
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Create unified encoder stack using the factory-based approach.

    Args:
        cfg: Configuration dictionary
        d_model: Model dimension
        n_heads: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        n_layers: Number of encoder layers

    Returns:
        Tuple of (encoder, pairwise_bias_module)
        - encoder: The configured encoder stack
        - pairwise_bias_module: PairwiseAttentionBias module if type is 'pa', None otherwise
    """
    from .unified_encoder import BatchedPairwiseProjection, SharedPairwiseProjection, UnifiedEncoderLayer, UnifiedTransformerEncoder

    # Parse architecture configuration
    arch_config = parse_architecture_config(cfg)
    validate_architecture_config(arch_config)

    # Create unified encoder layer
    encoder_layer = UnifiedEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, cfg=cfg, batch_first=True)

    # Create pairwise projection if PA is enabled
    pw_projection = None
    if arch_config["bias_type"] in ("pa", "alibi_pa"):
        r_pw = arch_config["bias_config"].get("pw_hidden_dim", 16)
        if arch_config["bias_config"].get("per_layer_pw", True):
            pw_projection = BatchedPairwiseProjection(r_pw, n_heads, n_layers)
        else:
            pw_projection = SharedPairwiseProjection(r_pw, n_heads)

    # Create encoder stack
    gradient_checkpointing = cfg.get("gradient_checkpointing", False)
    encoder = UnifiedTransformerEncoder(
        encoder_layer,
        num_layers=n_layers,
        gradient_checkpointing=gradient_checkpointing,
        pw_projection=pw_projection,
    )

    # Create external PA bias module (computes pairwise features once, shared across layers)
    pairwise_bias = None
    if arch_config["bias_type"] in ("pa", "alibi_pa"):
        pairwise_bias = create_relative_bias("pa", n_heads, d_model, arch_config["bias_config"])

    logger.debug(
        f"Unified encoder: pos_encoding={arch_config['pos_type']}, "
        f"relative_bias={arch_config['bias_type']}, "
        f"attn_backend={arch_config['attn_backend']}"
    )

    return encoder, pairwise_bias
