"""Unified encoder layer that uses factory pattern for component creation.

This module provides a single encoder layer that can handle all combinations of:
- Positional encodings (sinusoidal, RoPE, none)
- Relative biases (ALiBi, RPE, PA, none)
- Attention mechanisms (Flash, BiasAware, Standard)

This replaces the complex modular encoder with a cleaner, factory-based approach.
"""

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from instanovo_fm.model.attention.flash import FlashMHA

from .factories import (
    create_attention_mechanism,
    create_positional_encoding,
    create_relative_bias,
    parse_architecture_config,
    validate_architecture_config,
)


class UnifiedEncoderLayer(nn.TransformerEncoderLayer):
    """Unified encoder layer that supports all combinations of positional encoding and relative bias.

    This layer uses factory functions to create components based on configuration,
    making it much simpler and more maintainable than the previous modular approach.

    When PA bias is enabled, the per-layer g_pw projection is handled by the
    parent ``UnifiedTransformerEncoder`` via ``BatchedPairwiseProjection``.
    The pre-computed (B, H, L, L) bias for this layer is passed as
    ``pa_layer_bias`` instead of raw ``pairwise_feats``.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, cfg: Dict[str, Any], batch_first: bool = True) -> None:
        """Initialise the input."""
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)

        self.batch_first = batch_first

        # Parse and validate architecture configuration
        arch_config = parse_architecture_config(cfg)
        validate_architecture_config(arch_config)

        # Store configuration for debugging (use object.__setattr__ to avoid nn.Module restrictions)
        object.__setattr__(self, "pos_type", arch_config["pos_type"])
        object.__setattr__(self, "bias_type", arch_config["bias_type"])
        object.__setattr__(self, "attn_backend", arch_config["attn_backend"])
        object.__setattr__(self, "is_modular", arch_config["is_modular"])

        # Create positional encoding components
        self.pos_embed, self.rotary_emb = create_positional_encoding(arch_config["pos_type"], d_model, nhead, arch_config["pos_config"])

        # Create relative bias component
        # For PA type, the external PairwiseAttentionBias handles bias;
        # per-layer modules would only return zeros, so skip them.
        # For alibi_pa, create ALiBi as per-layer relative bias (PA is external).
        if arch_config["bias_type"] == "pa":
            self.relative_bias = None
        elif arch_config["bias_type"] == "alibi_pa":
            self.relative_bias = create_relative_bias("alibi", nhead, d_model, arch_config["bias_config"])
        else:
            self.relative_bias = create_relative_bias(arch_config["bias_type"], nhead, d_model, arch_config["bias_config"])

        # Create attention mechanism
        self.custom_attention = create_attention_mechanism(
            arch_config["bias_type"], d_model, nhead, dropout, self.rotary_emb, arch_config["attn_backend"]
        )

        # Replace attention if we have a custom one
        if self.custom_attention is not None:
            self.self_attn = self.custom_attention
            # Add batch_first attribute that nn.TransformerEncoder expects
            object.__setattr__(self.self_attn, "batch_first", batch_first)

        # NOTE: per-layer g_pw is NO LONGER created here.
        # It has been moved to BatchedPairwiseProjection in the encoder stack
        # for batched computation (single kernel launch for all layers).
        # Keep pw_norm/g_pw as None for backward compatibility with checkpoint
        # loading code that may check hasattr.
        self.pw_norm = None
        self.g_pw = None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        pairwise_feats: Optional[torch.Tensor] = None,
        pa_layer_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass with unified component handling.

        Args:
            src: Input tensor of shape (B, L, D)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
            attn_bias: Optional attention bias tensor
            pairwise_feats: Optional pairwise features tensor of shape (B, L, L, r).
                Only used when per-layer g_pw is on the layer itself (legacy path).
            pa_layer_bias: Pre-computed PA bias for this layer (B, H, L, L).
                Produced by BatchedPairwiseProjection in the encoder stack.
            is_causal: Whether to use causal attention
            return_attn_weights: Whether to return attention weights (forces manual computation for FlashMHA)

        Returns:
            Output tensor of shape (B, L, D)
        """
        # Store original sequence length for bias computation
        original_seq_len = src.size(1)

        # Note: sinusoidal PE is applied once in UnifiedTransformerEncoder.forward(),
        # not here. RoPE is applied per-layer inside the attention mechanism (correct).

        # Compute relative bias if available (use original sequence length)
        relative_bias = None
        if self.relative_bias is not None:
            relative_bias = self.relative_bias.get_bias(original_seq_len, src.device, src.dtype)

        # Use pre-computed PA layer bias from BatchedPairwiseProjection.
        # Fall back to per-layer g_pw only for legacy checkpoints that still
        # have g_pw weights on the layer itself.
        layer_bias = pa_layer_bias
        if layer_bias is None and self.g_pw is not None and pairwise_feats is not None:
            layer_bias = self.g_pw(self.pw_norm(pairwise_feats)).permute(0, 3, 1, 2)

        # Combine biases: attn_bias (external/padded), relative_bias (ALiBi/RPE), layer_bias (PA)
        parts = [b for b in (attn_bias, relative_bias, layer_bias) if b is not None]
        if parts:
            combined_bias = parts[0]
            for b in parts[1:]:
                combined_bias = combined_bias + b
        else:
            combined_bias = None

        # Handle different attention types
        if hasattr(self.self_attn, "qkv"):
            # Custom attention mechanism (BiasAwareMHA, FlashMHA, etc.)
            # These handle RoPE and relative bias internally
            if isinstance(self.self_attn, FlashMHA):
                # FlashMHA can accept attn_bias when bias is used (falls back to manual computation)
                # Also pass return_attn_weights to force manual computation when weights are needed
                src2, _ = self.self_attn(
                    src,
                    attn_mask=src_mask,
                    attn_bias=combined_bias,  # Pass combined bias for attention weight extraction
                    key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                    return_attn_weights=return_attn_weights,  # Force manual computation when weights needed
                )
            else:
                # BiasAwareMHA or other custom attention
                src2, _ = self.self_attn(
                    src,
                    attn_mask=src_mask,
                    is_causal=is_causal,
                    attn_bias=combined_bias,  # Pass combined bias
                    key_padding_mask=src_key_padding_mask,
                )
        else:
            # Standard nn.MultiheadAttention
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal)

        # Standard transformer layer operations
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BatchedPairwiseProjection(nn.Module):
    """Pre-compute all per-layer PA biases in a single batched operation.

    Instead of N separate ``LayerNorm → Linear(r_pw, n_heads)`` calls (one per
    encoder layer), this module applies LayerNorm once and then a single
    ``Linear(r_pw, n_heads * n_layers)`` projection, turning N kernel launches
    into 1.

    The output is sliced per-layer in the encoder loop.

    Args:
        r_pw: Pairwise feature dimension (hidden_dim of PairwiseAttentionBias).
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
    """

    # With r_pw=64, H=12, N=9, L=200: output is (B, L, L, 108) bf16.
    # At B=512 that is ~8.4 GB — enough to OOM on a single H100.
    # Process in chunks of 64 to cap peak memory at ~1 GB per chunk.
    _CHUNK_SIZE = 64

    def __init__(self, r_pw: int, n_heads: int, n_layers: int) -> None:
        """Initialise the input."""
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pw_norm = nn.LayerNorm(r_pw)
        # Single Linear projects to all layers at once: r_pw → n_heads * n_layers
        self.g_pw_batched = nn.Linear(r_pw, n_heads * n_layers, bias=False)

    def forward(self, pairwise_feats: torch.Tensor) -> list[torch.Tensor]:
        """Compute per-layer PA biases in one shot.

        Args:
            pairwise_feats: (B, L, L, r_pw)

        Returns:
            List of n_layers tensors, each (B, n_heads, L, L).
        """
        B = pairwise_feats.shape[0]  # noqa: N806
        # Wrap each chunk in gradient checkpointing during training.
        # Without it, LayerNorm + Linear saves accumulate ~0.61 GB per chunk
        # across all 8 chunks. Checkpointing keeps only one chunk's activations
        # live at a time during the backward pass.
        run_chunk = (
            (lambda pw: torch.utils.checkpoint.checkpoint(self._forward_chunk, pw, use_reentrant=False)) if self.training else self._forward_chunk
        )
        if B <= self._CHUNK_SIZE:
            return run_chunk(pairwise_feats)
        chunks = [run_chunk(pairwise_feats[i : i + self._CHUNK_SIZE]) for i in range(0, B, self._CHUNK_SIZE)]
        # chunks is a list of lists; zip to concatenate per-layer
        return [torch.cat([c[l] for c in chunks], dim=0) for l in range(self.n_layers)]  # noqa: E741

    def _forward_chunk(self, pairwise_feats: torch.Tensor) -> list[torch.Tensor]:
        normed = self.pw_norm(pairwise_feats)  # (C, L, L, r_pw)
        all_biases = self.g_pw_batched(normed)  # (C, L, L, H*N)
        C, L1, L2, _ = all_biases.shape  # noqa: N806
        all_biases = all_biases.view(C, L1, L2, self.n_layers, self.n_heads)
        all_biases = all_biases.permute(3, 0, 4, 1, 2)  # (N, C, H, L, L)
        return list(all_biases.unbind(0))


class SharedPairwiseProjection(nn.Module):
    """Project pairwise features to attention bias once, shared across all layers.

    A single ``LayerNorm → Linear(r_pw, n_heads)`` produces one (B, H, L, L)
    bias tensor that is reused by every encoder layer.  This is 9× cheaper
    than ``BatchedPairwiseProjection`` (output is n_heads instead of
    n_heads × n_layers) at the cost of less per-layer specialisation.

    Args:
        r_pw: Pairwise feature dimension (hidden_dim of PairwiseAttentionBias).
        n_heads: Number of attention heads.
    """

    def __init__(self, r_pw: int, n_heads: int) -> None:
        """Initialise the input."""
        super().__init__()
        self.n_heads = n_heads
        self.pw_norm = nn.LayerNorm(r_pw)
        self.g_pw = nn.Linear(r_pw, n_heads, bias=False)

    def forward(self, pairwise_feats: torch.Tensor) -> list[torch.Tensor]:
        """Compute a single shared PA bias.

        Args:
            pairwise_feats: (B, L, L, r_pw)

        Returns:
            List with one element (for API compatibility): [(B, n_heads, L, L)].
            The encoder broadcasts this to all layers.
        """
        normed = self.pw_norm(pairwise_feats)  # (B, L, L, r_pw)
        bias = self.g_pw(normed)  # (B, L, L, H)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        return [bias]


class UnifiedTransformerEncoder(nn.Module):
    """Unified transformer encoder that stacks UnifiedEncoderLayer layers.

    This encoder supports all combinations of positional encoding and relative bias
    through the unified layer implementation.
    """

    def __init__(
        self,
        encoder_layer: UnifiedEncoderLayer,
        num_layers: int,
        gradient_checkpointing: bool = False,
        pw_projection: Optional["BatchedPairwiseProjection"] = None,
    ) -> None:
        """Initialise the input."""
        super().__init__()

        # Lift sinusoidal pos_embed from the layer to the encoder stack.
        # Sinusoidal PE must be applied ONCE before the layer loop, not per-layer.
        # RoPE is unaffected — it's applied inside attention in each layer (correct).
        self.pos_embed = encoder_layer.pos_embed
        encoder_layer.pos_embed = None  # Remove from template before deep-copy

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = copy.deepcopy(encoder_layer.norm1)
        self.gradient_checkpointing = gradient_checkpointing

        # Batched PA projection (None when PA is not enabled)
        self.pw_projection = pw_projection

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        pairwise_feats: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            src: Input tensor of shape (B, L, D)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
            attn_bias: Optional attention bias tensor
            pairwise_feats: Optional pairwise features tensor of shape (B, L, L, r)
            is_causal: Whether to use causal attention
            return_attn_weights: Whether to return attention weights (forces manual computation for FlashMHA)

        Returns:
            Output tensor of shape (B, L, D)
        """
        output = src

        # Apply sinusoidal positional encoding once before the layer loop
        if self.pos_embed is not None:
            batch_first = self.layers[0].batch_first if self.layers else True
            if batch_first:
                output = output.transpose(0, 1)  # (B, L, D) -> (L, B, D)
            output = self.pos_embed(output)
            if batch_first:
                output = output.transpose(0, 1)  # (L, B, D) -> (B, L, D)

        # Pre-compute PA biases (batched per-layer or shared across layers)
        pa_layer_biases: list[Optional[torch.Tensor]] | None = None
        if self.pw_projection is not None and pairwise_feats is not None:
            pa_layer_biases = self.pw_projection(pairwise_feats)

        for i, layer in enumerate(self.layers):
            # SharedPairwiseProjection returns a single-element list;
            # BatchedPairwiseProjection returns one element per layer.
            if pa_layer_biases is not None:
                pa_bias_i = pa_layer_biases[min(i, len(pa_layer_biases) - 1)]
            else:
                pa_bias_i = None

            if self.gradient_checkpointing and self.training:
                output = torch.utils.checkpoint.checkpoint(
                    layer,
                    output,
                    src_mask,
                    src_key_padding_mask,
                    attn_bias,
                    None,  # pairwise_feats not needed when pa_bias_i is provided
                    pa_bias_i,
                    is_causal,
                    return_attn_weights,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    attn_bias=attn_bias,
                    pa_layer_bias=pa_bias_i,
                    is_causal=is_causal,
                    return_attn_weights=return_attn_weights,
                )
        return self.norm(output)
