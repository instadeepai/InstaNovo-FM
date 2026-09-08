"""
Attention mechanisms for the InstaNovo model.

This module provides various attention implementations:
- Flash attention for efficient computation
- Bias-aware attention with pairwise attention (PA) support
- Base attention utilities and classes
"""

from .flash import FlashMHA
from .bias_aware import BiasAwareMHA
from .base import BaseAttention

__all__ = [
    "FlashMHA",
    "BiasAwareMHA",
    "BaseAttention",
]
