"""
Relative Positional Encoding (RPE) implementation.

This module provides Shaw's Relative Positional Encoding which uses learned
embeddings for relative positions between tokens.
"""

import torch
import torch.nn as nn
from typing import Optional


class RelativePositionalEncoding(nn.Module):
    """Shaw's Relative Positional Encoding (RPE) implementation."""
    
    def __init__(self, max_relative_position: int, d_model: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.relative_attention_bias = nn.Embedding(2 * max_relative_position + 1, d_model)
    
    def forward(self, length: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Create relative position bias matrix.
        
        Args:
            length: Sequence length
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tensor of shape (L, L) with relative position embeddings
        """
        # Create relative position indices
        range_vec = torch.arange(length, device=device, dtype=torch.long)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.T
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get embeddings
        embeddings = self.relative_attention_bias(final_mat)
        
        # Convert dtype if needed
        if dtype is not None:
            embeddings = embeddings.to(dtype=dtype)
        
        return embeddings
    
    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get relative position bias tensor for the given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype
            
        Returns:
            Relative position bias tensor of shape (1, 1, L, L)
        """
        embeddings = self.forward(seq_len, device, dtype)
        # Project to scalar bias: (L, L, d_model) -> (L, L)
        bias = torch.sum(embeddings, dim=-1)  # Simple projection
        return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L) 