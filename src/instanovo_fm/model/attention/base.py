"""
Base attention classes and utilities.

This module provides base classes and utilities for attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional


class BaseAttention(nn.Module):
    """
    Base class for attention mechanisms.

    This provides a common interface for different attention implementations
    and can be extended for specific attention types.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.dropout = dropout

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for attention mechanism.

        Args:
            query: Query tensor of shape (B, L, D)
            key: Key tensor of shape (B, L, D)
            value: Value tensor of shape (B, L, D)
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            is_causal: Whether to use causal attention

        Returns:
            Tuple of (output, attention_weights)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def _scaled_dot_product_attention(self,
                                     query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     attn_mask: Optional[torch.Tensor] = None,
                                     key_padding_mask: Optional[torch.Tensor] = None,
                                     is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (B, H, L, D)
            key: Key tensor of shape (B, H, L, D)
            value: Value tensor of shape (B, H, L, D)
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            is_causal: Whether to use causal attention

        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, float('-inf'))

        # Apply key padding mask
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))

        # Apply causal mask if requested
        if is_causal:
            causal_mask = torch.triu(torch.ones(scores.size(-1), scores.size(-1),
                                               device=scores.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask, float('-inf'))

        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.dropout(attention_weights, self.dropout, self.training)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
