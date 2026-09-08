"""
Sinusoidal positional encoding implementation.

This module provides the standard sinusoidal positional encoding used in the original Transformer paper.
"""

import math
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[Tensor, "token batch embedding"]
    ) -> Float[Tensor, "token batch embedding"]:
        """Positional encoding forward pass.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # x has shape [seq_len, batch_size, embedding_dim]
        # self.pe has shape [1, max_len, embedding_dim]
        # We want to add positional encoding for the first seq_len positions
        seq_len = x.size(0)
        # Transpose PE to match x shape: [1, seq_len, d_model] -> [seq_len, 1, d_model]
        # Then broadcast to [seq_len, batch_size, d_model]
        pe_slice = self.pe[:, :seq_len].transpose(0, 1)  # [seq_len, 1, d_model]
        x = x + pe_slice
        return self.dropout(x)
