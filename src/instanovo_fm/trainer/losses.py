"""
Loss functions for foundational pre-training.

This module contains all loss computation logic for masked m/z prediction,
auxiliary tasks (charge classification, retention time regression), and
variance regularization.
"""

from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn.functional as F

from instanovo_fm.trainer.utils import (
    mu_law_encode,
    mu_law_decode,
    mz_to_bin_groups,
    focal_loss,
)


def mask_invalid_offsets(
    offset_logits: torch.Tensor,
    group_pred: torch.Tensor,
    n_groups: int,
    last_group_size: int,
) -> torch.Tensor:
    """
    Mask invalid offset logits for the last bin group.
    
    The last group may have fewer valid bins than group_size. This function
    sets logits for invalid bins to -inf to prevent them from being selected.
    
    Args:
        offset_logits: Offset logits of shape (M, group_size)
        group_pred: Group predictions of shape (M,)
        n_groups: Total number of groups
        last_group_size: Number of valid bins in last group
        
    Returns:
        Masked offset logits of shape (M, group_size)
        
    Example:
        >>> offset_logits = torch.randn(100, 100)  # 100 samples, 100 bins per group
        >>> group_pred = torch.randint(0, 125, (100,))  # Some in last group (124)
        >>> masked = mask_invalid_offsets(offset_logits, group_pred, 125, 50)
        >>> # Samples with group_pred == 124 now have logits[:, 50:] = -inf
    """
    # Find samples in last group
    is_last = (group_pred == (n_groups - 1))
    
    if is_last.any():
        # Clone to avoid in-place modification
        offset_logits = offset_logits.clone()
        
        # Set invalid logits to -inf
        m_idx = is_last.nonzero(as_tuple=True)[0]
        offset_logits[m_idx, last_group_size:] = -float("inf")
    
    return offset_logits


def compute_classifier_confidence(
    group_logits: torch.Tensor,
    offset_logits: torch.Tensor,
    n_groups: int,
    last_group_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-peak top-1 probability for the group and offset classification
    heads, plus their product as the joint confidence.

    Args:
        group_logits: Group logits of shape (B, L, n_groups)
        offset_logits: Offset logits of shape (B, L, group_size)
        n_groups: Total number of groups
        last_group_size: Number of valid bins in last group

    Returns:
        Dictionary with:
            - conf_group: Group top-1 probability (B, L)
            - conf_offset: Offset top-1 probability (B, L), computed after
              masking invalid offsets in the last (partial) group
            - conf_joint: conf_group * conf_offset (B, L)
    """
    B, L, _ = group_logits.shape

    # Mask invalid offsets in the last (partial) group before softmax, so
    # conf_offset reflects only physically valid bins.
    group_pred = group_logits.argmax(dim=-1)  # (B, L)
    offset_logits_flat = offset_logits.reshape(-1, offset_logits.size(-1))
    group_pred_flat = group_pred.reshape(-1)
    offset_logits_masked_flat = mask_invalid_offsets(
        offset_logits_flat, group_pred_flat, n_groups, last_group_size
    )
    offset_logits_masked = offset_logits_masked_flat.reshape(B, L, -1)

    conf_group = F.softmax(group_logits, dim=-1).max(dim=-1).values
    conf_offset = F.softmax(offset_logits_masked, dim=-1).max(dim=-1).values
    conf_joint = conf_group * conf_offset

    return {
        "conf_group": conf_group,
        "conf_offset": conf_offset,
        "conf_joint": conf_joint,
    }





def compute_classification_loss(
    preds: Tuple[torch.Tensor, torch.Tensor],
    target_mz: torch.Tensor,
    mask: torch.Tensor,
    config: Dict,
    group_size: int,
    n_groups: int,
    last_group_size: int,
    bin_edges: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classification loss for m/z prediction using group + offset heads.

    Returns: (total_loss, classification_loss, group_loss, offset_loss)
    """
    import torch.nn.functional as F

    mz_head_cfg      = config.get("mz_head", {})
    binning_cfg      = mz_head_cfg.get("binning", {})
    bin_size         = binning_cfg.get("bin_size", mz_head_cfg.get("bin_size", None))
    max_mz           = config.get("max_mz", 2500.0)
    min_mz           = config.get("min_mz", 0.0)

    w_group          = mz_head_cfg.get("w_group", 0.5)
    w_offset         = mz_head_cfg.get("w_offset", 0.5)

    # SORD: Gaussian soft ordinal labels (Diaz & Marathe, CVPR 2019) for the offset
    # head. 0.0 = disabled (standard one-hot CE).
    sord_sigma = mz_head_cfg.get("sord_sigma", 0.0)

    group_logits, offset_logits = preds  # [B,L,G], [B,L,K]
    device = group_logits.device

    target_mz_scaled = target_mz * max_mz  # [B, L] in Da

    # Filter mask: some rare LCFM spectra have padding/zero-mass peaks that can
    # end up in the mask. Exclude them from loss computation rather than crash.
    if mask.dtype != torch.bool:
        mask = mask.bool()
    mask = mask & (target_mz_scaled >= min_mz)

    target_groups, target_offsets = mz_to_bin_groups(
        target_mz_scaled, bin_size, max_mz, group_size, min_mz, bin_edges=bin_edges
    )

    if not mask.any():
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero, zero, zero

    b_idx_all, l_idx_all = mask.nonzero(as_tuple=True)
    glog = group_logits[b_idx_all, l_idx_all]           # [M, G]
    olog = offset_logits[b_idx_all, l_idx_all]          # [M, K]
    tg   = target_groups[b_idx_all, l_idx_all]          # [M]
    to   = target_offsets[b_idx_all, l_idx_all]         # [M]

    # Group loss: standard hard cross-entropy
    group_loss = F.cross_entropy(glog, tg, reduction="mean")

    # Offset loss — mask invalid offsets for the last group first
    olog = mask_invalid_offsets(olog, tg, n_groups, last_group_size)
    is_last_t = (tg == (n_groups - 1))

    if sord_sigma > 0.0:
        offsets_range = torch.arange(group_size, device=to.device).unsqueeze(0).float()  # (1, K)
        distances = offsets_range - to.unsqueeze(1).float()  # (M, K)
        soft_target = torch.exp(-distances ** 2 / (2 * sord_sigma ** 2))  # (M, K)

        if is_last_t.any():
            invalid_mask = offsets_range.long() >= last_group_size  # (1, K)
            soft_target[is_last_t] = soft_target[is_last_t].masked_fill(invalid_mask, 0.0)

        soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        offset_log_probs = F.log_softmax(olog, dim=-1)
        safe_log_probs = offset_log_probs.nan_to_num(nan=0.0, neginf=0.0)
        offset_loss = -(soft_target * safe_log_probs).sum(dim=1).mean()
    else:
        offset_loss = F.cross_entropy(olog, to, reduction="mean")

    classification_loss = w_group * group_loss + w_offset * offset_loss
    total_loss = classification_loss
    return total_loss, classification_loss, group_loss, offset_loss


def compute_ptm_loss(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    gamma: float = 2.0, 
    alpha: float = 0.25
) -> torch.Tensor:
    """
    Compute focal loss for PTM binary classification.
    
    Args:
        logits: PTM logits of shape (B, 2)
        labels: PTM labels of shape (B,) with values 0 or 1
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        
    Returns:
        PTM classification loss
    """
    # Use focal loss for better handling of class imbalance
    return focal_loss(logits, labels, gamma=gamma, alpha=alpha)


def compute_dmz_loss(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    gamma: float = 2.0, 
    alpha: float = 0.25
) -> torch.Tensor:
    """
    Compute focal loss for Δm/z classification.
    Ignores -100 labels.
    """
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    valid = labels_flat != -100
    if not valid.any():
        # Return a zero loss that is connected to the computation graph if possible
        zero_loss = logits.sum() * 0.0
        if not zero_loss.requires_grad:
            zero_loss = torch.zeros((), device=logits.device, requires_grad=True)
        return zero_loss
    logits_valid = logits_flat[valid]
    labels_valid = labels_flat[valid]
    return focal_loss(logits_valid, labels_valid, gamma=gamma, alpha=alpha)


# ===============================================
# Intensity Regression Loss
# ===============================================
#
# Why Huber Loss for Proteomic Intensity?
#
# Proteomic MS/MS intensity data has several characteristics that make
# robust regression essential:
#
# 1. Outliers are Common:
#    - Chimeric spectra (multiple peptides in one scan)
#    - Instrument artifacts (electronic noise, detector saturation)
#    - Isotope pattern variations (natural abundance fluctuations)
#    - Neutral losses (H₂O, NH₃) with variable intensity
#
# 2. MSE is Too Sensitive:
#    - Quadratic penalty: (y - ŷ)² grows rapidly for large errors
#    - Outliers dominate gradient updates
#    - Model learns to fit outliers at expense of typical data
#
# 3. Huber Loss Balances Robustness and Efficiency:
#    - L2 penalty for errors < delta: Smooth gradients near optimum
#    - L1 penalty for errors > delta: Bounded influence of outliers
#    - Delta parameter controls transition point
#
# 4. Why delta=0.1 for Intensity?
#    - Intensities are √transformed and L2-normalized
#    - Typical values in [0, 0.5] range
#    - 10% error threshold reasonable for proteomic intensities
#    - Balance: smaller delta = more robust, larger = more efficient


def compute_intensity_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 0.1
) -> torch.Tensor:
    """
    Compute Huber loss for intensity regression.

    Huber loss is more robust to outliers than MSE, critical for proteomic
    intensity prediction where chimeric spectra, instrument artifacts, and
    isotope variations introduce unexpected peaks.

    The loss transitions from L2 (MSE) for errors < delta to L1 (MAE) for
    larger errors, preventing outliers from dominating gradients.

    Args:
        predictions: Intensity predictions (B, L, 1) in [0, max_intensity]
        targets: Ground-truth intensities (B, L), -100 for ignored positions
        delta: Huber threshold (0.1 for L2-normalized intensities)

    Returns:
        Scalar Huber loss (zero with gradient if no valid positions)
    """
    # Flatten
    pred_flat = predictions.view(-1)  # (B*L,)
    target_flat = targets.view(-1)    # (B*L,)

    # Filter out ignored positions (-100)
    valid = target_flat != -100
    if not valid.any():
        # Return a zero loss that is connected to the computation graph if possible
        zero_loss = predictions.sum() * 0.0
        if not zero_loss.requires_grad:
            zero_loss = torch.zeros((), device=predictions.device, requires_grad=True)
        return zero_loss

    pred_valid = pred_flat[valid]     # (N_valid,)
    target_valid = target_flat[valid] # (N_valid,)

    # Compute Huber loss (smooth_l1_loss with beta=delta)
    # PyTorch smooth_l1_loss implements Huber loss:
    # Loss = 0.5 * (x - y)² / beta,  if |x - y| < beta
    #      = |x - y| - 0.5 * beta,    otherwise
    return F.smooth_l1_loss(pred_valid, target_valid, beta=delta, reduction="mean")


def compute_auxiliary_losses(
    aux_out: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    aux_config: Dict,
    model_config: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute auxiliary task losses (charge classification, retention time regression, Δm/z classification, PTM classification, and intensity regression).
    
    Args:
        aux_out: Auxiliary outputs from model
        batch: Input batch containing targets
        aux_config: Auxiliary task configuration
        
    Returns:
        Tuple of (charge_loss, rt_loss, dmz_loss, ptm_loss, intensity_loss, total_aux_loss)
    """
    # Lambdas control compute and weighting
    lambda_charge = aux_config.get("lambda_charge", 0.0)
    lambda_rt = aux_config.get("lambda_rt", 0.0)
    lambda_dmz = aux_config.get("lambda_dmz", 0.0)
    lambda_ptm = aux_config.get("lambda_ptm", 0.0)
    lambda_intensity = aux_config.get("lambda_intensity", 0.0)

    # Get device from aux_out tensors - use a more robust approach
    device = None
    if aux_out and isinstance(aux_out, dict):
        for key, value in aux_out.items():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        # If no tensor found, try to get from latent
        if device is None and "latent" in aux_out:
            device = aux_out["latent"].device
        # If still no device, try to get from batch tensors
        if device is None and batch:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break

    # Charge classification loss (only if enabled)
    if lambda_charge > 0.0 and "charge" in aux_out:
        charge_loss = F.cross_entropy(aux_out["charge"], batch["charge_id"])
    else:
        charge_loss = torch.zeros((), device=device, requires_grad=True)
    
    # RT loss removed (MDN dead code)
    rt_loss = torch.zeros((), device=device, requires_grad=True)
    
    # Δm/z classification loss (only if enabled)
    if lambda_dmz > 0.0 and "dmz" in aux_out and "dmz_labels" in batch:
        dmz_loss = compute_dmz_loss(
            aux_out["dmz"], 
            batch["dmz_labels"], 
            gamma=aux_config.get("dmz_focal_gamma", 2.0),
            alpha=aux_config.get("dmz_focal_alpha", 0.25)
        )
    else:
        dmz_loss = torch.zeros((), device=device, requires_grad=True)
    
    # PTM classification loss (only if enabled)
    if lambda_ptm > 0.0 and "ptm" in aux_out and "ptm_present" in batch:
        ptm_loss = compute_ptm_loss(
            aux_out["ptm"],
            batch["ptm_present"],
            gamma=aux_config.get("ptm_focal_gamma", 2.0),
            alpha=aux_config.get("ptm_focal_alpha", 0.25)
        )
    else:
        ptm_loss = torch.zeros((), device=device, requires_grad=True)
    
    # Create intensity_labels from spectra if not present
    # This handles the case where intensity task is enabled but labels weren't created in data processor
    if lambda_intensity > 0.0 and "intensity" in aux_out:
        if "intensity_labels" not in batch:
            # Extract intensity targets from spectra
            targets = batch.get("spectra")  # (B, L, 2) [m/z, intensity]
            if targets is not None:
                intensity_targets = targets[:, :, 1]  # (B, L) - L2-normalized intensities

                # Create labels with -100 for invalid positions
                intensity_labels = torch.full_like(intensity_targets, -100.0)

                # Valid mask: not padding (spectra_mask=False)
                # Note: We'll apply mlm_mask restriction later in the existing code
                if "spectra_mask" in batch:
                    valid_mask = ~batch["spectra_mask"]
                    intensity_labels[valid_mask] = intensity_targets[valid_mask]
                else:
                    intensity_labels = intensity_targets.clone()

                batch["intensity_labels"] = intensity_labels

    # Intensity regression loss (only if enabled)
    intensity_head_cfg = model_config.get("intensity_head", {})
    if lambda_intensity > 0.0 and "intensity" in aux_out and "intensity_labels" in batch:
        # Restrict intensity learning to masked, non-padding tokens to avoid leakage from visible intensities
        intensity_labels = batch["intensity_labels"]
        valid_mask = intensity_labels != -100
        if "mlm_mask" in batch:
            valid_mask = valid_mask & batch["mlm_mask"].bool()
        if "spectra_mask" in batch:
            valid_mask = valid_mask & (~batch["spectra_mask"].bool())

        # Convert mask to label ignore by setting non-valid to -100
        masked_intensity_labels = torch.where(valid_mask, intensity_labels, torch.full_like(intensity_labels, -100))

        intensity_loss = compute_intensity_regression_loss(
            aux_out["intensity"],
            masked_intensity_labels,
            delta=intensity_head_cfg.get("huber_delta", 0.1)
        )
    else:
        intensity_loss = torch.zeros((), device=device, requires_grad=True)
    
    # Total loss with configurable weights
    total_aux_loss = (
        lambda_charge * charge_loss
        + lambda_rt * rt_loss
        + lambda_dmz * dmz_loss
        + lambda_ptm * ptm_loss
        + lambda_intensity * intensity_loss
    )
    
    return charge_loss, rt_loss, dmz_loss, ptm_loss, intensity_loss, total_aux_loss


def compute_total_loss(
    mz_loss: torch.Tensor,
    aux_loss: torch.Tensor,
) -> torch.Tensor:
    """
    Combine all loss components into total loss.

    Args:
        mz_loss: Main m/z prediction loss
        aux_loss: Auxiliary task loss

    Returns:
        Total loss
    """
    return mz_loss + aux_loss
