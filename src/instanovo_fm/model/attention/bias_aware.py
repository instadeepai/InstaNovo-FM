import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BiasAwareMHA(nn.Module):
    """
    General-purpose multi-head attention module that can handle various types of attention bias.
    
    This attention mechanism supports:
    - Standard multi-head attention
    - Rotary Position Embeddings (RoPE)
    - Arbitrary attention bias (PA, ALiBi, RPE, etc.)
    - Key padding masks
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
        rotary_emb: Optional rotary embedding for RoPE
        
    Note:
        - attn_bias must be of shape (B, H, L, L) where B=batch, H=heads, L=sequence_length
        - key_padding_mask must be boolean tensor of shape (B, L) where True indicates pad positions
    """
    def __init__(self, embed_dim, num_heads, dropout=0., rotary_emb=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.p = dropout  # Store dropout value for logging
        
        # Let the accelerator manage dtype casting automatically
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.o   = nn.Linear(embed_dim, embed_dim)
        self.dp  = nn.Dropout(dropout)
        self.rotary_emb = rotary_emb

    # Exclude from torch.compile — the dynamic bias shapes cause a Triton
    # codegen bug (incompatible dimensions in online_softmax_reduce).
    @torch.compiler.disable
    def forward(
        self,
        x,
        attn_mask=None,          # ignored
        attn_bias=None,          # used
        key_padding_mask=None,
        is_causal=False,         # ignored
        **_,
    ):
        """
        Forward pass with bias-aware attention.
        
        Args:
            x: Input tensor of shape (B, L, D)
            attn_mask: Ignored (for compatibility)
            attn_bias: Optional attention bias of shape (B, H, L, L)
            key_padding_mask: Optional padding mask of shape (B, L)
            is_causal: Ignored (for compatibility)
            
        Returns:
            Tuple of (output, attention_weights) where attention_weights has shape (B, H, L, L)
        """
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE if available and enabled
        if (self.rotary_emb is not None and 
            hasattr(self.rotary_emb, "rotary_pct") and 
            self.rotary_emb.rotary_pct > 0):
            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k)

        # Compute scaled attention scores: QK^T / sqrt(d)
        dots = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)
        dots = dots / math.sqrt(self.d)

        # Add bias after scaling (post-scale convention: AF2, AF3, ALiBi, T5, SDPA)
        if attn_bias is not None:
            # Handle broadcastable bias (e.g., from cache with batch size 1)
            if attn_bias.shape[0] == 1 and attn_bias.shape[0] != q.shape[0]:
                attn_bias = attn_bias.expand(q.shape[0], -1, -1, -1)
            dots = dots + attn_bias                       # (B,H,L,L)
        
        if key_padding_mask is not None:
            dots.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

        att = torch.softmax(dots, -1)
        att = self.dp(att)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, L, -1)
        return self.o(y), att  # Return full multi-head attention weights instead of mean 