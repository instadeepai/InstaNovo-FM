"""
PadTokenMixin for handling pad token replacement in Flash Attention models.

This mixin provides a clean, reusable way to replace zero-padded spectra with
learned pad tokens, which is required for Flash Attention compatibility.
"""

import torch
import torch.nn as nn
from typing import Optional


class PadTokenMixin:
    """
    Mixin for handling pad token replacement in Flash Attention models.
    
    Flash Attention requires non-zero embeddings for all positions, so we replace
    zero-padded spectra with learned pad tokens. This mixin provides a clean,
    reusable implementation of this logic.
    
    Usage:
        class MyModel(nn.Module, PadTokenMixin):
            def __init__(self, cfg):
                super().__init__()
                self.setup_pad_token(cfg)
            
            def forward(self, x):
                x = self.apply_pad_token_replacement(x)
                # ... rest of forward pass
    """
    
    def setup_pad_token(self, cfg: dict, dim_model: int):
        """Setup pad token based on attention backend configuration.
        
        Flash Attention requires non-zero embeddings at all positions, so we use
        learned pad tokens instead of attention masks for padding.
        
        Args:
            cfg: Configuration dictionary
            dim_model: Model dimension for pad token
        """
        # Determine attention backend from architecture config
        attn_config = cfg.get('architecture', {}).get('attention', {})
        attn_backend = attn_config.get("backend", "math")
        self.use_flash_attention = (attn_backend == "flash")
        
        if self.use_flash_attention:
            # Create pad token parameter for Flash Attention backward-compat.
            #
            # Under the fixed encoder (src_key_padding_mask is always passed
            # through, see Encoder Contract in instanovo/foundational/CLAUDE.md
            # §4), padded keys are set to -inf in every softmax, so this
            # parameter CANNOT influence any attention output — it is inert.
            # We keep it so old flash-trained checkpoints load cleanly via
            # strict=False, but freeze it (requires_grad=False) so training
            # does not accidentally revive the historical leak by letting the
            # pad_token pick up a non-zero learned contribution.
            self.pad_token = nn.Parameter(
                torch.randn(1, 1, dim_model), requires_grad=False
            )
        else:
            # No pad token needed for standard attention
            self.pad_token = None
    
    def apply_pad_token_replacement(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Replace `x[b,i]` with `self.pad_token` wherever `pad_mask[b,i]==True`.
        Args:
            x: Embedded spectra tensor of shape (B, L, D)
            pad_mask: Boolean tensor of shape (B, L), True where pad token should be used
        Returns:
            Updated x tensor with pad tokens replacing pad_mask positions
        """
        if not self.use_flash_attention or self.pad_token is None:
            # No pad token replacement needed
            return x
        
        # Guard against shape mismatch to catch caller mistakes
        if pad_mask.shape[1] != x.shape[1]:
            raise ValueError(f"pad_mask shape {pad_mask.shape} doesn't match x shape {x.shape} in sequence dimension")
        
        # Only replace positions where pad_mask is True
        if pad_mask.any():
            # Create pad embedding with correct shape for broadcasting
            pad_embed = self.pad_token.expand(x.size(0), x.size(1), -1)
            x = torch.where(pad_mask.unsqueeze(-1), pad_embed, x)
        
        return x
    
    def get_pad_token(self) -> Optional[nn.Parameter]:
        """
        Get the pad token parameter if it exists.
        
        Returns:
            Pad token parameter or None if not using Flash Attention
        """
        return self.pad_token if hasattr(self, 'pad_token') else None 