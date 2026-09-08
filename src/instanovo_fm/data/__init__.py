"""Data processing module for the InstaNovo Foundation Model.

This module provides the FoundationalDataProcessor class for self-supervised
learning on mass spectrometry data without peptide sequence annotations.
"""

from __future__ import annotations

from instanovo_fm.data.data import FoundationalDataProcessor
from instanovo_fm.data.masking import (
    find_isotopic_neighbors,
    get_mask_function,
    thompson_sampling_mask,
    thompson_sampling_span_mask,
    uniform_random_mask,
)

__all__ = [
    "FoundationalDataProcessor",
    "thompson_sampling_mask",
    "thompson_sampling_span_mask",
    "uniform_random_mask",
    "find_isotopic_neighbors",
    "get_mask_function",
]
