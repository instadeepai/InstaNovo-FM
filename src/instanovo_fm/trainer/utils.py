"""
Utility functions for the foundational model.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def mu_law_encode(x: torch.Tensor, k: int = 255) -> torch.Tensor:
    """
    μ-law encoding for compressing m/z values.

    Args:
        x: Input tensor in [0, 1] range
        k: Compression parameter (default: 255)

    Returns:
        Encoded tensor in [-1, 1] range
    """
    # Ensure x is in [0, 1] range
    x = torch.clamp(x, 0.0, 1.0)

    # Center the signal to [-1, 1] range for symmetric μ-law encoding
    y = 2 * x - 1  # Map [0, 1] to [-1, 1]

    # Apply μ-law encoding
    mu = torch.tensor(k - 1, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(y) * torch.log1p(mu * torch.abs(y)) / torch.log1p(mu)

    return x_mu


def mu_law_decode(x_mu: torch.Tensor, k: int = 255) -> torch.Tensor:
    """
    μ-law decoding for decompressing m/z values.

    Args:
        x_mu: Encoded tensor in [-1, 1] range
        k: Compression parameter (default: 255)

    Returns:
        Decoded tensor in [0, 1] range
    """
    # Ensure x_mu is in [-1, 1] range
    x_mu = torch.clamp(x_mu, -1.0, 1.0)

    # Apply μ-law decoding
    mu = torch.tensor(k - 1, dtype=x_mu.dtype, device=x_mu.device)
    y = torch.sign(x_mu) * (torch.exp(torch.abs(x_mu) * torch.log1p(mu)) - 1) / mu

    # Map back from [-1, 1] to [0, 1] range
    x = (y + 1) / 2

    return torch.clamp(x, 0.0, 1.0)


def mz_to_bins(
    mz_values: torch.Tensor,
    bin_size: Optional[float] = None,
    max_mz: Optional[float] = None,
    min_mz: float = 0.0,
    bin_edges: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert m/z values to bin indices for classification.

    Supports both uniform (bin_size) and non-uniform (bin_edges) binning.

    Args:
        mz_values: m/z values in Da (can be scaled or unscaled)
        bin_size: Size of each bin in Da (for uniform binning)
        max_mz: Maximum m/z value (for uniform binning)
        min_mz: Minimum m/z value (default: 0.0)
        bin_edges: Bin edges tensor [n_bins+1] for non-uniform binning

    Returns:
        Bin indices (0-based, relative to min_mz)
    """
    if bin_edges is not None:
        # Non-uniform binning: use binary search
        edges_device = bin_edges.to(mz_values.device)
        # Use right=True so mz == edges[k] returns k+1, then -1 gives bin k
        # This ensures values at left edge of bin k go to bin k (not k-1)
        bin_indices = torch.searchsorted(edges_device, mz_values, right=True) - 1
        n_bins = len(bin_edges) - 1
        return torch.clamp(bin_indices, 0, n_bins - 1)
    else:
        # Uniform binning: use formula (fast path)
        if bin_size is None or max_mz is None:
            raise ValueError("bin_size and max_mz required for uniform binning")

        # Convert to bin indices relative to min_mz
        bin_indices = torch.floor((mz_values - min_mz) / bin_size).long()

        # Clamp to valid range [0, n_bins-1]
        n_bins = int((max_mz - min_mz) / bin_size) + 1
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)

        return bin_indices


def bins_to_mz(
    bin_indices: torch.Tensor,
    bin_size: Optional[float] = None,
    min_mz: float = 0.0,
    bin_edges: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert bin indices back to m/z values (center of bin).

    Supports both uniform (bin_size) and non-uniform (bin_edges) binning.

    Args:
        bin_indices: Bin indices (0-based, relative to min_mz)
        bin_size: Size of each bin in Da (for uniform binning)
        min_mz: Minimum m/z value (default: 0.0)
        bin_edges: Bin edges tensor [n_bins+1] for non-uniform binning

    Returns:
        m/z values (center of each bin)
    """
    if bin_edges is not None:
        # Non-uniform binning: use bin edges to compute centers
        edges_device = bin_edges.to(bin_indices.device)
        left = edges_device[bin_indices]
        right = edges_device[bin_indices + 1]
        return (left + right) / 2.0
    else:
        # Uniform binning: use formula (fast path)
        if bin_size is None:
            raise ValueError("bin_size required for uniform binning")
        return min_mz + (bin_indices.float() + 0.5) * bin_size


def mz_to_bin_groups(
    mz_values: torch.Tensor,
    bin_size: Optional[float] = None,
    max_mz: Optional[float] = None,
    group_size: int = None,
    min_mz: float = 0.0,
    bin_edges: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert m/z values to bin group indices and offsets.

    Supports both uniform (bin_size) and non-uniform (bin_edges) binning.

    Args:
        mz_values: m/z values in Da
        bin_size: Size of each bin in Da (for uniform binning)
        max_mz: Maximum m/z value (for uniform binning)
        group_size: Number of bins per group
        min_mz: Minimum m/z value (default: 0.0)
        bin_edges: Bin edges tensor [n_bins+1] for non-uniform binning

    Returns:
        Tuple of (group_indices, offset_indices)
    """
    # Convert to bin indices (supports both uniform and non-uniform)
    bin_indices = mz_to_bins(mz_values, bin_size, max_mz, min_mz, bin_edges)

    # Calculate group and offset
    group_indices = bin_indices // group_size
    offset_indices = bin_indices % group_size

    return group_indices, offset_indices


def bin_groups_to_mz(
    group_indices: torch.Tensor,
    offset_indices: torch.Tensor,
    bin_size: Optional[float] = None,
    group_size: int = None,
    min_mz: float = 0.0,
    bin_edges: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert bin group indices and offsets back to m/z values.

    Supports both uniform (bin_size) and non-uniform (bin_edges) binning.

    Args:
        group_indices: Group indices
        offset_indices: Offset indices within groups
        bin_size: Size of each bin in Da (for uniform binning)
        group_size: Number of bins per group
        min_mz: Minimum m/z value (default: 0.0)
        bin_edges: Bin edges tensor [n_bins+1] for non-uniform binning

    Returns:
        m/z values
    """
    bin_indices = group_indices * group_size + offset_indices
    return bins_to_mz(bin_indices, bin_size, min_mz, bin_edges)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Focal Loss for handling class imbalance.

    Args:
        logits: Model predictions (N, num_classes) - flattened batch
        targets: Ground truth labels (N,) - flattened batch of class indices
        gamma: Focusing parameter (default: 2.0)
        alpha: Weighting parameter for rare classes (default: 0.25)

    Returns:
        Focal loss value
    """
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Get the probability of the correct class
    num_classes = logits.shape[-1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    pt = (probs * targets_one_hot).sum(dim=-1)  # (N,)

    # Calculate focal loss with improved class weighting
    focal_weight = (1 - pt) ** gamma

    # Use per-class alpha weighting: alpha for positive class, 1-alpha for negative
    alpha_weight = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
    alpha_weight = alpha_weight.sum(dim=-1)  # (N,)

    focal_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)

    return focal_loss.mean()
