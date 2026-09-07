from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class RtRegHead(nn.Module):
    """Regression head for retention time prediction."""
    
    def __init__(self, d_model: int, max_log_rt: float = 10.5):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)   # ½ width keeps params tiny
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.max = max_log_rt                        # register as constant

        # Kaiming-uniform ≈ xavier for GELU
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B"]:
        """
        Forward pass for RT regression.
        
        Args:
            x: Input tensor of shape (B, D)
            
        Returns:
            RT predictions of shape (B,)
        """
        y = self.fc2(self.act(self.fc1(x)))            # linear range
        y = torch.tanh(y) * self.max                   # bound to ±max_log_rt
        return y.squeeze(-1)                           # (B,)


class MDNRtHead(nn.Module):
    """
    Mixture-Density Network head for retention time prediction.
    
    Predicts K-component Gaussian mixture for log-RT:
      • mu       : (B,K)   – means, scaled to ±max_log_rt
      • log_sigma: (B,K)   – unconstrained log‐σ
      • log_pi   : (B,K)   – log mixture weights (log-softmax)
    """
    
    def __init__(self, d_model: int, n_components: int = 3, max_log_rt: float = 10.5):
        super().__init__()
        self.n_components = n_components
        self.max = max_log_rt

        # simple 2-layer MLP → mixture params
        hidden = d_model // 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden, n_components)      # un-scaled
        self.log_sigma = nn.Linear(hidden, n_components)      # any real
        self.log_pi = nn.Linear(hidden, n_components)      # logits

        # init like RtRegHead
        for layer in [self.mu, self.log_sigma, self.log_pi]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)

    def forward(self, x: Float[torch.Tensor, "B D"]) -> Dict[str, Float[torch.Tensor, "B K"]]:
        """
        Forward pass for MDN RT prediction.
        
        Args:
            x: Input tensor of shape (B, D)
            
        Returns:
            Dictionary with mixture parameters:
            - mu: means of shape (B, K)
            - log_sigma: log standard deviations of shape (B, K)
            - log_pi: log mixture weights of shape (B, K)
        """
        h = self.mlp(x)
        mu = torch.tanh(self.mu(h)) * self.max         # bound means
        log_sigma = self.log_sigma(h).clamp(-7.0, 7.0)        # avoid under/overflow
        log_pi = torch.log_softmax(self.log_pi(h), dim=-1)
        return {"mu": mu, "log_sigma": log_sigma, "log_pi": log_pi}


class MzRegressionHead(nn.Module):
    """Regression head for m/z prediction with μ-law scaling."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L 1"]:
        """
        Forward pass for m/z regression.

        Args:
            x: Input tensor of shape (B, L, D)

        Returns:
            M/z predictions of shape (B, L, 1) in μ-law encoded space
        """
        return self.head(x)


class MzHeteroscedasticRegressionHead(nn.Module):
    """Heteroscedastic regression head for m/z prediction with μ-law scaling.

    Predicts both mean (μ) and log-variance (log σ²) in μ-law encoded space.
    The mean output is unbounded linear (clamped at decode time).
    The log-variance enables per-peak uncertainty estimation via heteroscedastic
    Gaussian NLL loss.
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Mean head: predicts μ in μ-law space (unbounded, clamped at decode)
        self.mu_head = nn.Linear(d_model, 1)

        # Log-variance head: predicts log(σ²) in μ-law space
        self.log_var_head = nn.Linear(d_model, 1)

        # Initialization
        nn.init.xavier_uniform_(self.shared[0].weight)
        nn.init.zeros_(self.shared[0].bias)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        # Initialize log_var to predict neutral variance:
        # log(σ²) = 0.0 → σ² = 1.0, neutral starting point
        nn.init.xavier_uniform_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(
        self, x: Float[torch.Tensor, "B L D"]
    ) -> Tuple[Float[torch.Tensor, "B L 1"], Float[torch.Tensor, "B L 1"]]:
        """Forward pass for heteroscedastic m/z regression.

        Args:
            x: Input tensor of shape (B, L, D)

        Returns:
            Tuple of (mu, log_var) where:
            - mu: Mean predictions in μ-law space (unbounded), shape (B, L, 1)
            - log_var: Log-variance predictions, shape (B, L, 1)
        """
        h = self.shared(x)
        mu = self.mu_head(h)  # (B, L, 1) unbounded
        log_var = self.log_var_head(h)  # (B, L, 1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


class MzClassificationHead(nn.Module):
    """
    Classification head for m/z prediction using grouped bins.

    Splits the mass range into groups and predicts:
    1. Group index (which bin group)
    2. Offset within group (which bin within the group)

    Supports optional offset conditioning where the offset head receives
    group information via an embedding lookup. During training with
    teacher forcing, the true group is used; during inference, the
    predicted group (argmax) is used.
    """

    def __init__(
        self,
        d_model: int,
        n_bin_groups: int,
        bin_group_size: int,
        offset_conditioning: str = "none",
        group_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_bin_groups = n_bin_groups
        self.bin_group_size = bin_group_size
        self.offset_conditioning = offset_conditioning

        # Group prediction head
        self.group_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_bin_groups),
        )

        # Offset prediction head (within group)
        if offset_conditioning != "none":
            self.group_embedding = nn.Embedding(n_bin_groups, group_embed_dim)
            offset_input_dim = d_model + group_embed_dim
        else:
            self.group_embedding = None
            offset_input_dim = d_model

        self.offset_head = nn.Sequential(
            nn.Linear(offset_input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, bin_group_size),
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B L D"],
        target_groups: Optional[torch.Tensor] = None,
    ) -> Tuple[Float[torch.Tensor, "B L G"], Float[torch.Tensor, "B L O"]]:
        """
        Forward pass for m/z classification.

        Args:
            x: Input tensor of shape (B, L, D)
            target_groups: Optional ground-truth group indices (B, L) for teacher
                forcing. When provided (training), the offset head is conditioned
                on these. When None (inference), the predicted group (argmax) is used.

        Returns:
            Tuple of (group_logits, offset_logits) where:
            - group_logits: shape (B, L, n_bin_groups) - logits for bin group prediction
            - offset_logits: shape (B, L, bin_group_size) - logits for offset within group
        """
        group_logits = self.group_head(x)  # (B, L, n_bin_groups)

        if self.offset_conditioning == "none":
            offset_logits = self.offset_head(x)  # (B, L, bin_group_size)
        else:
            # Teacher forcing: true groups in training, predicted in inference
            if target_groups is not None:
                group_idx = target_groups
            else:
                group_idx = group_logits.argmax(dim=-1)

            # No .detach() — embedding learns from offset loss gradient.
            # Integer index lookup has no gradient path to group_head.
            group_embed = self.group_embedding(group_idx)  # (B, L, group_embed_dim)
            offset_input = torch.cat([x, group_embed], dim=-1)
            offset_logits = self.offset_head(offset_input)  # (B, L, bin_group_size)

        return group_logits, offset_logits


class ChargeHead(nn.Module):
    """Classification head for charge state prediction."""
    
    def __init__(self, d_model: int, n_charge_classes: int = 12):
        super().__init__()
        self.head = nn.Linear(d_model, n_charge_classes)
        
        # Initialize with proper weights
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B C"]:
        """
        Forward pass for charge classification.
        
        Args:
            x: Input tensor of shape (B, D)
            
        Returns:
            Charge logits of shape (B, n_charge_classes)
        """
        return self.head(x)


class PTMHead(nn.Module):
    """Binary classification head for PTM presence prediction."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),  # Binary classification: 0=no PTM, 1=PTM present
        )
        
        # Initialize with proper weights
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B 2"]:
        """
        Forward pass for PTM classification.
        
        Args:
            x: Input tensor of shape (B, D)
            
        Returns:
            PTM logits of shape (B, 2) for binary classification
        """
        return self.head(x)


class DeltaMzHead(nn.Module):
    """
    Classification head for Δm/z prediction using chemical gap buckets.
    
    Predicts which chemical gap bucket lies between a peak i and its next k peaks (k = 1 ... 7).
    This is a self-supervised auxiliary task that helps the model learn chemical relationships.
    """
    
    def __init__(self, d_model: int, n_buckets: int = 8):
        super().__init__()
        self.n_buckets = n_buckets
        
        # Simple 2-layer MLP for Δm/z classification
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_buckets),
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L C"]:
        """
        Forward pass for Δm/z classification.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Δm/z bucket logits of shape (B, L, n_buckets)
        """
        return self.head(x)


class IntensityRegressionHead(nn.Module):
    """
    Regression head for intensity prediction.
    
    Predicts continuous intensity values. Since intensity distributions are typically
    skewed with more low-intensity peaks, this head uses a transformation that can
    handle this skew effectively.
    """
    
    def __init__(self, d_model: int, max_intensity: float = 1.0):
        super().__init__()
        self.max_intensity = max_intensity
        
        # 2-layer MLP for intensity regression
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L 1"]:
        """
        Forward pass for intensity regression.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Intensity predictions of shape (B, L, 1) in range [0, max_intensity]
        """
        # Get raw predictions
        raw_pred = self.head(x)
        
        # Apply sigmoid to ensure positive values in [0, 1], then scale to max_intensity
        # This handles the skew towards low intensities naturally
        intensity_pred = torch.sigmoid(raw_pred) * self.max_intensity
        
        return intensity_pred


class PredictionHeads(nn.Module):
    """
    Container for all prediction heads used in the InstaNovo encoder.
    
    This class manages all the different prediction heads and provides
    a unified interface for forward passes.
    """
    
    def __init__(
        self,
        d_model: int,
        mz_task: str = "regression",
        n_bin_groups: Optional[int] = None,
        bin_group_size: Optional[int] = None,
        n_charge_classes: int = 12,
        n_rt_components: int = 3,
        max_log_rt: float = 10.5,
        aux_enabled: bool = True,
        charge_enabled: bool = False,
        rt_enabled: bool = False,
        dmz_enabled: bool = False,
        ptm_enabled: bool = False,
        intensity_enabled: bool = False,
        max_intensity: float = 1.0,
        heteroscedastic: bool = False,
        offset_conditioning: str = "none",
        group_embed_dim: int = 64,
    ):
        super().__init__()
        self.heteroscedastic = heteroscedastic

        # M/Z prediction heads
        if mz_task == "classification":
            if n_bin_groups is None or bin_group_size is None:
                raise ValueError("n_bin_groups and bin_group_size must be provided for classification task")
            self.mz_head = MzClassificationHead(
                d_model, n_bin_groups, bin_group_size,
                offset_conditioning=offset_conditioning,
                group_embed_dim=group_embed_dim,
            )
        else:  # regression
            if heteroscedastic:
                self.mz_head = MzHeteroscedasticRegressionHead(d_model)
            else:
                self.mz_head = MzRegressionHead(d_model)
        
        # Auxiliary prediction heads (strictly gated by per-task flags)
        self.aux_enabled = aux_enabled
        self.charge = ChargeHead(d_model, n_charge_classes=n_charge_classes) if aux_enabled and charge_enabled else None
        self.rt_mdn = MDNRtHead(d_model, n_components=n_rt_components, max_log_rt=max_log_rt) if aux_enabled and rt_enabled else None
        
        # Δm/z prediction head (only if enabled)
        self.dmz_enabled = dmz_enabled
        if dmz_enabled:
            self.dmz_head = DeltaMzHead(d_model, n_buckets=8)
        else:
            self.dmz_head = None
        
        # PTM prediction head (only if enabled)
        self.ptm_enabled = ptm_enabled
        if ptm_enabled:
            self.ptm_head = PTMHead(d_model)
        else:
            self.ptm_head = None
        
        # Intensity prediction head (only if enabled, regression only)
        self.intensity_enabled = intensity_enabled
        if intensity_enabled:
            self.intensity_head = IntensityRegressionHead(d_model, max_intensity=max_intensity)
        else:
            self.intensity_head = None

        # Store configuration
        self.mz_task = mz_task

    @staticmethod
    def _bin_conditioning(
        bin_edges: torch.Tensor,
        bin_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bin centers and widths from edges and flat bin indices.

        Args:
            bin_edges: 1-D tensor of bin edges [n_bins + 1] in Daltons.
            bin_indices: Integer tensor of any shape with flat bin indices.

        Returns:
            Tuple of (bin_centers, bin_widths) with same shape as bin_indices.
        """
        edges = bin_edges.to(bin_indices.device)
        left = edges[bin_indices]
        right = edges[bin_indices + 1]
        return (left + right) / 2.0, right - left

    # Exclude prediction heads from torch.compile — they are tiny 2-layer MLPs
    # where compilation gives negligible speedup. The encoder stack (attention,
    # FFN, layer norms) is where 99%+ of the compile benefit comes from.
    # Compiling heads causes SIGSEGV due to: (1) self.training guards triggering
    # recompilation on every train↔eval switch, (2) nn.Embedding + torch.cat in
    # the offset conditioning path, (3) variable-key dict construction in aux_out.
    @torch.compiler.disable
    def forward(
        self,
        x_head: Float[torch.Tensor, "B L D"],
        latent: Float[torch.Tensor, "B D"],
        target_groups: Optional[torch.Tensor] = None,
        target_offsets: Optional[torch.Tensor] = None,
        bin_edges: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass through all prediction heads.

        Args:
            x_head: Input tensor for m/z prediction of shape (B, L, D)
            latent: Latent token for auxiliary predictions of shape (B, D)
            target_groups: Optional true group indices (B, L) for teacher forcing
                in the offset head. If None, uses argmax of group logits at inference.
            target_offsets: Optional true offset indices (B, L) for teacher forcing
                the delta-ppm head. Used with target_groups to compute GT bin index.
            bin_edges: Optional 1-D bin edges tensor [n_bins+1] in Daltons.
                Required for conditioned delta-ppm during training.

        Returns:
            Tuple of (mz_predictions, auxiliary_outputs) where:
            - mz_predictions:
              * For regression: (B, L, 1) tensor with m/z predictions in [-1,1] range
              * For classification: tuple of (group_logits, offset_logits)
            - auxiliary_outputs: Dict with charge, RT, Δm/z, PTM, and intensity predictions
              * intensity: (B, L, 1) for regression
        """
        # M/Z predictions
        reg_log_var = None
        if self.mz_task == "classification":
            mz_preds = self.mz_head(x_head, target_groups=target_groups)
            # Add logits to aux_out for entropy metrics
            group_logits, offset_logits = mz_preds
        else:  # regression
            if self.heteroscedastic:
                mu, log_var = self.mz_head(x_head)
                mz_preds = mu  # Primary prediction is the mean
                reg_log_var = log_var
            else:
                mz_preds = self.mz_head(x_head)
            group_logits, offset_logits = None, None
        
        # Auxiliary predictions (using latent token)
        if self.aux_enabled:
            aux_out = {
                "charge": self.charge(latent) if self.charge is not None else torch.zeros(latent.size(0), 1, device=latent.device, requires_grad=True),
                "rt_mdn": self.rt_mdn(latent) if self.rt_mdn is not None else {
                    "mu": torch.zeros(latent.size(0), 1, device=latent.device, requires_grad=True),
                    "log_sigma": torch.zeros(latent.size(0), 1, device=latent.device, requires_grad=True),
                    "log_pi": torch.zeros(latent.size(0), 1, device=latent.device, requires_grad=True)
                },  # Returns dict with mu, log_sigma, log_pi
                "latent": latent,  # expose for variance regularisation
            }
        else:
            # Return dummy outputs when auxiliary tasks are disabled
            device = latent.device
            aux_out = {
                "charge": torch.zeros(latent.size(0), 1, device=device, requires_grad=True),  # Dummy charge output
                "rt_mdn": {
                    "mu": torch.zeros(latent.size(0), 1, device=device, requires_grad=True),
                    "log_sigma": torch.zeros(latent.size(0), 1, device=device, requires_grad=True),
                    "log_pi": torch.zeros(latent.size(0), 1, device=device, requires_grad=True)
                },  # Dummy RT output
                "latent": latent,  # expose for variance regularisation
            }
        
        # Δm/z predictions (using sequence tokens)
        if self.dmz_enabled and self.dmz_head is not None:
            aux_out["dmz"] = self.dmz_head(x_head)  # (B, L, n_buckets)
        
        # PTM predictions (using latent token)
        if self.ptm_enabled and self.ptm_head is not None:
            aux_out["ptm"] = self.ptm_head(latent)  # (B, 2)
        
        # Intensity predictions (using sequence tokens)
        if self.intensity_enabled and self.intensity_head is not None:
            aux_out["intensity"] = self.intensity_head(x_head)  # (B, L, n_bins) or (B, L, 1)

        # Heteroscedastic regression: store log-variance for loss and confidence
        if reg_log_var is not None:
            aux_out["reg_log_var"] = reg_log_var

        # Add classification logits to aux_out for entropy metrics (classification only)
        if self.mz_task == "classification" and group_logits is not None and offset_logits is not None:
            aux_out["group_logits"] = group_logits
            aux_out["offset_logits"] = offset_logits

        # Store encoder tokens for top-k recomputation (eval only — skip during
        # training to avoid retaining the full (B, L, D) tensor through backward)
        if (
            not self.training
            and self.mz_task == "classification"
            and self.mz_head.offset_conditioning != "none"
        ):
            aux_out["x_tokens"] = x_head

        return mz_preds, aux_out
