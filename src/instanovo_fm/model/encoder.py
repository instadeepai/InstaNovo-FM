"""Foundation Model for InstaNovo.

Self-supervised learning on MS/MS spectra through masked peak reconstruction.
This is an encoder-only transformer that learns spectral representations without
requiring peptide annotations.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo_fm.model.embeddings import (
    DualPeakEmbedding,
    FourierPeakEmbedding,
    LinearPeakEmbedding,
    MetaTokenEmbed,
    MultiScalePeakEmbedding,
    RBFPeakEmbedding,
)
from instanovo_fm.model.encoder_layers.factories import (
    create_unified_encoder_stack,
)
from instanovo_fm.model.heads import PredictionHeads
from instanovo_fm.model.ion_ladder import IonLadderEncoder
from instanovo_fm.model.pad_token_mixin import PadTokenMixin
from instanovo_fm.model.pairwise_bias import PairwiseAttentionBias
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.file_downloader import download_file

MODEL_TYPE = "foundational"

logger = ColorLog(console, __name__).logger


class FoundationModel(nn.Module, PadTokenMixin):
    """Foundation model for self-supervised learning on MS/MS spectra.

    Encoder-only transformer architecture that learns spectral representations
    through masked m/z reconstruction. Unlike supervised models (InstaNovo,
    InstaNovo+), this model doesn't require peptide annotations during training.

    Key features:
        - Encoder-only architecture (no decoder)
        - Masked peak reconstruction task
        - Optional auxiliary tasks (RT, charge, Δm/z prediction)
        - Supports both regression and classification for m/z prediction
        - Flash Attention and Pairwise Attention (PA) bias support
        - Meta token for precursor/metadata encoding

    Args:
        dim_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        dim_feedforward: Feedforward dimension (default: 2048)
        n_layers: Number of encoder layers (default: 6)
        dropout: Dropout rate (default: 0.1)
        n_peaks: Maximum number of peaks (default: 200)
        max_mz: Maximum m/z value (default: 2500.0)
        min_mz: Minimum m/z value (default: 0.0)
        max_charge: Maximum charge state (default: 10)
        peak_encoder_type: Peak encoder type: "multiscale", "fourier", "rbf", or "linear" (default: "multiscale")
        mz_task: M/z prediction task: "regression" or "classification" (default: "regression")
        use_meta_token: Use meta token for metadata (default: True)
        cfg: Configuration dictionary containing architecture settings including attention backend
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_peaks: int = 200,
        max_mz: float = 2500.0,
        min_mz: float = 0.0,
        max_charge: int = 10,
        peak_encoder_type: str = "multiscale",
        mz_task: str = "regression",
        use_meta_token: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the foundation model."""
        super().__init__()

        # Store configuration
        self.cfg = cfg or {}
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_peaks = n_peaks
        self.max_mz = max_mz
        self.min_mz = min_mz
        self.max_charge = max_charge
        self.peak_encoder_type = peak_encoder_type
        self.mz_task = mz_task
        self.use_meta_token = use_meta_token

        # Setup pad token and determine attention backend from config
        # Sets self.use_flash_attention based on architecture.attention.backend
        self.setup_pad_token(self.cfg, self.dim_model)

        # ===== Peak Encoder =====
        if self.peak_encoder_type == "fourier":
            fourier_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_freqs = fourier_config.get("num_freqs", dim_model // 4)
            strategy = fourier_config.get("strategy", "voronov_et_al")
            trainable_fourier = fourier_config.get("trainable_fourier", True)
            x_min = fourier_config.get("x_min", 0.001)
            x_max = fourier_config.get("x_max", 1.5)
            self.peak_encoder = FourierPeakEmbedding(
                dim_model,
                dropout=dropout,
                num_freqs=num_freqs,
                strategy=strategy,
                trainable_fourier=trainable_fourier,
                x_min=x_min,
                x_max=x_max,
            )
        elif self.peak_encoder_type == "rbf":
            rbf_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_rbf = rbf_config.get("num_rbf", 1024)
            normalize_mz = self.cfg.get("normalize_mz", True)
            self.peak_encoder = RBFPeakEmbedding(
                dim_model,
                dropout=dropout,
                min_mz=min_mz,
                max_mz=max_mz,
                num_rbf=num_rbf,
                normalize_mz=normalize_mz,
            )
        elif self.peak_encoder_type == "dual":
            dual_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_rbf = dual_config.get("num_rbf", 2048)
            normalize_mz = self.cfg.get("normalize_mz", True)
            self.peak_encoder = DualPeakEmbedding(
                dim_model,
                dropout=dropout,
                min_mz=min_mz,
                max_mz=max_mz,
                num_rbf=num_rbf,
                normalize_mz=normalize_mz,
            )
        elif self.peak_encoder_type == "linear":
            self.peak_encoder = LinearPeakEmbedding(dim_model, dropout=dropout)
        else:
            self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)

        # ===== Ion Ladder Encoder (optional) =====
        ion_ladder_cfg = self.cfg.get("ion_ladder", {})
        self.ion_ladder_enabled = ion_ladder_cfg.get("enabled", False)
        if self.ion_ladder_enabled:
            residue_masses_dict = ion_ladder_cfg.get("residue_masses", {})
            if not residue_masses_dict:
                raise ValueError("ion_ladder.enabled=True but no residue_masses provided. Ensure residue masses are injected into model config.")
            residue_masses_list = list(residue_masses_dict.values())
            # Build neutral losses dict if enabled
            neutral_losses: dict[str, float] | None = None
            nl_cfg = ion_ladder_cfg.get("neutral_losses", {})
            if nl_cfg.get("enabled", False):
                loss_types = nl_cfg.get("loss_types", ["H2O", "NH3"])
                neutral_losses = {name: IonLadderEncoder.KNOWN_LOSSES[name] for name in loss_types if name in IonLadderEncoder.KNOWN_LOSSES}
                if not neutral_losses:
                    raise ValueError(f"neutral_losses.enabled=True but none of {loss_types} are known. Known: {list(IonLadderEncoder.KNOWN_LOSSES)}")

            self.ion_ladder = IonLadderEncoder(
                d_model=dim_model,
                residue_masses=residue_masses_list,
                window_k=ion_ladder_cfg.get("window_k", 40),
                sigma=ion_ladder_cfg.get("sigma", 0.35),
                charge_states=tuple(ion_ladder_cfg.get("charge_states", [1, 2])),
                neutral_losses=neutral_losses,
            )
            n_ref = len(self.ion_ladder.residue_masses_buf)
            logger.debug(
                f"IonLadder: {len(residue_masses_list)} residues, {n_ref} ref masses, "
                f"k={ion_ladder_cfg.get('window_k', 40)}, charges={ion_ladder_cfg.get('charge_states', [1, 2])}"
            )
        else:
            self.ion_ladder = None  # type: ignore[assignment]

        # ===== Special Tokens =====
        self.latent_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim_model) * 0.02)

        # ===== Meta Token (optional) =====
        if self.use_meta_token:
            self.meta_embed = self._setup_meta_token()

        # ===== Encoder Stack =====
        self.encoder, self.pairwise_bias = self._setup_encoder()

        # ===== Prediction Heads =====
        self.prediction_heads = self._setup_prediction_heads()

        # ===== Noise Configuration =====
        noise_config = self.cfg.get("noise", {})
        self.mz_noise_ppm = noise_config.get("mz_noise_ppm", 0.0)
        self.intensity_noise_pct = noise_config.get("intensity_noise_pct", 0.0)
        self.intensity_noise_floor = noise_config.get("intensity_noise_floor", 0.001)

        # ===== Gaussian-Blurred Mask Encoding =====
        masking_config = self.cfg.get("masking", {})
        self.blur_sigma_da = masking_config.get("blur_sigma_da", 0.0)
        if self.blur_sigma_da > 0.0:
            logger.info(f"Gaussian-blurred mask encoding: sigma={self.blur_sigma_da} Da")

        # ===== Visible Intensity for Masked Peaks =====
        self.mask_intensity = masking_config.get("mask_intensity", True)
        if not self.mask_intensity and self.blur_sigma_da <= 0.0:
            logger.warning("mask_intensity=False requires blur_sigma_da > 0; falling back to masked intensity")
            self.mask_intensity = True
        if not self.mask_intensity:
            logger.info("Visible intensity: masked peaks retain true intensity")

        # ===== Blurred PA Bias =====
        bias_cfg = self.cfg.get("architecture", {}).get("relative_bias", {}).get("config", {})
        self.blur_masked_pa = bias_cfg.get("blur_masked_pa", False)
        if self.blur_masked_pa and self.blur_sigma_da <= 0.0:
            logger.warning("blur_masked_pa=True but blur_sigma_da=0.0; blurred PA disabled")
            self.blur_masked_pa = False
        if self.blur_masked_pa and self.pairwise_bias is None:
            logger.warning("blur_masked_pa=True but PA not enabled; blurred PA disabled")
            self.blur_masked_pa = False

        # ===== Mask-to-Mask PA Bias =====
        self.pa_mask_to_mask = bias_cfg.get("pa_mask_to_mask", False)
        if self.pa_mask_to_mask and self.pairwise_bias is None:
            logger.warning("pa_mask_to_mask=True but PA not enabled; disabled")
            self.pa_mask_to_mask = False
        if self.pa_mask_to_mask:
            logger.info("Mask-to-mask PA: keeping exact PA between masked peaks")

        # Get attention backend for logging
        attn_backend = self.cfg.get("architecture", {}).get("attention", {}).get("backend", "math")

        logger.info(f"FoundationModel initialized: dim_model={dim_model}, n_layers={n_layers}, mz_task={mz_task}, attention_backend={attn_backend}")

    def _setup_meta_token(self) -> MetaTokenEmbed:
        """Setup meta token embedding for precursor/metadata encoding."""
        meta_cfg = self.cfg.get("meta_token", {})

        # Get vocabulary sizes from metadata builder
        from instanovo_fm.data.metadata_builder import get_vocabulary_sizes

        vocab_sizes = get_vocabulary_sizes()

        # Extract field inclusion flags from config
        include_frag_type = meta_cfg.get("include_frag_type", True)
        include_instrument = meta_cfg.get("include_instrument", True)
        include_acquisition = meta_cfg.get("include_acquisition", True)
        include_detector = meta_cfg.get("include_detector", True)
        include_enzyme = meta_cfg.get("include_enzyme", True)
        include_quant = meta_cfg.get("include_quant", True)
        include_precursor_charge = meta_cfg.get("include_precursor_charge", True)
        include_precursor_mass = meta_cfg.get("include_precursor_mass", True)
        include_collision_energy = meta_cfg.get("include_collision_energy", True)

        return MetaTokenEmbed(
            proj_dim=self.dim_model,
            n_freq=meta_cfg.get("n_freq", 16),
            # Vocabulary sizes from metadata builder
            n_frag=vocab_sizes["n_frag"],
            n_instrument=vocab_sizes["n_instrument"],
            n_acquisition=vocab_sizes["n_acquisition"],
            n_detector=vocab_sizes["n_detector"],
            n_enzyme=vocab_sizes["n_enzyme"],
            n_quant=vocab_sizes["n_quant"],
            n_charge=vocab_sizes["n_charge"],
            # Continuous feature bounds
            precursor_mass_max=meta_cfg.get("precursor_mass_max", 10000.0),
            # Field inclusion
            include_frag_type=include_frag_type,
            include_instrument=include_instrument,
            include_acquisition=include_acquisition,
            include_detector=include_detector,
            include_enzyme=include_enzyme,
            include_quant=include_quant,
            include_precursor_charge=include_precursor_charge,
            include_precursor_mass=include_precursor_mass,
            include_collision_energy=include_collision_energy,
        )

    def _setup_encoder(self) -> Tuple[nn.Module, Optional[PairwiseAttentionBias]]:
        """Setup encoder stack using unified factory."""
        return create_unified_encoder_stack(  # type: ignore[no-any-return]
            cfg=self.cfg,
            d_model=self.dim_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            n_layers=self.n_layers,
        )

    def _setup_prediction_heads(self) -> PredictionHeads:
        """Setup all prediction heads."""
        # M/Z task configuration
        mz_head_cfg = self.cfg.get("mz_head", {})

        if self.mz_task == "classification":
            # Create binning strategy using factory
            from instanovo_fm.trainer.binning import create_binning_strategy

            self.binning_strategy = create_binning_strategy(self.cfg, self.min_mz, self.max_mz)

            # Register parameters as buffers for checkpoint persistence
            self.register_buffer("bin_group_size_tensor", torch.tensor(self.binning_strategy.bin_group_size))
            self.register_buffer("n_bins_tensor", torch.tensor(self.binning_strategy.n_bins))
            self.register_buffer("n_bin_groups_tensor", torch.tensor(self.binning_strategy.n_groups))
            self.register_buffer("last_group_size_tensor", torch.tensor(self.binning_strategy.last_group_size))

            # For non-uniform strategies, register bin edges as buffer
            bin_edges = self.binning_strategy.bin_edges
            if len(bin_edges) > 0:
                self.register_buffer("bin_edges", bin_edges)

            # Extract values for PredictionHeads initialization
            n_bin_groups = self.binning_strategy.n_groups
            bin_group_size = self.binning_strategy.bin_group_size
        else:
            n_bin_groups = None
            bin_group_size = None

        # Auxiliary task configuration
        aux_config = self.cfg.get("auxiliary", {})
        aux_enabled = aux_config.get("enabled", False)

        # Per-task enable flags from lambdas
        lambda_charge = aux_config.get("lambda_charge", 0.0)
        lambda_rt = aux_config.get("lambda_rt", 0.0)
        lambda_dmz = aux_config.get("lambda_dmz", 0.0)
        lambda_ptm = aux_config.get("lambda_ptm", 0.0)
        lambda_intensity = aux_config.get("lambda_intensity", 0.0)

        charge_enabled = aux_enabled and (lambda_charge > 0.0)
        rt_enabled = aux_enabled and (lambda_rt > 0.0)
        dmz_enabled = aux_enabled and (lambda_dmz > 0.0)
        ptm_enabled = aux_enabled and (lambda_ptm > 0.0)
        intensity_enabled = aux_enabled and (lambda_intensity > 0.0)

        # Intensity head configuration
        intensity_head_cfg = self.cfg.get("intensity_head", {})
        max_intensity = intensity_head_cfg.get("max_intensity", 1.0)

        # Heteroscedastic regression (only for regression task)
        heteroscedastic = mz_head_cfg.get("heteroscedastic", False) if self.mz_task != "classification" else False

        # Offset conditioning config
        offset_conditioning = mz_head_cfg.get("offset_conditioning", "none")
        group_embed_dim = mz_head_cfg.get("group_embed_dim", 64)

        return PredictionHeads(
            d_model=self.dim_model,
            mz_task=self.mz_task,
            n_bin_groups=n_bin_groups,
            bin_group_size=bin_group_size,
            n_charge_classes=aux_config.get("n_charge_classes", 10) if aux_enabled else 0,
            n_rt_components=aux_config.get("n_rt_components", 3) if aux_enabled else 0,
            max_log_rt=aux_config.get("max_log_rt", 10.5),
            aux_enabled=aux_enabled,
            charge_enabled=charge_enabled,
            rt_enabled=rt_enabled,
            dmz_enabled=dmz_enabled,
            ptm_enabled=ptm_enabled,
            intensity_enabled=intensity_enabled,
            max_intensity=max_intensity,
            heteroscedastic=heteroscedastic,
            offset_conditioning=offset_conditioning,
            group_embed_dim=group_embed_dim,
        )

    def _embed_peaks(self, spectra: torch.Tensor) -> torch.Tensor:
        """Embed peaks using configured peak encoder.

        Args:
            spectra: Input spectra (B, L, 2) [m/z, intensity]

        Returns:
            Peak embeddings (B, L, D)
        """
        return self.peak_encoder(spectra)

    def _apply_ion_ladder(
        self,
        x: torch.Tensor,
        spectra: torch.Tensor,
        spectra_mask: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply IonLadderEncoder to peak embeddings (if enabled).

        Must be called AFTER _embed_peaks and BEFORE _apply_mlm_mask.
        The mlm_mask is passed through so that masked peaks are excluded
        from neighbour windows — preventing an unmasked peak from seeing
        the m/z position of a masked neighbour (which would leak the
        reconstruction target via residue-mass matching).

        Features computed for masked positions are harmless: they will be
        discarded when _apply_mlm_mask replaces those embeddings with the
        [MASK] token.

        Args:
            x: Peak embeddings (B, L, D)
            spectra: Raw spectra (B, L, 2) [m/z, intensity]
            spectra_mask: Optional padding mask (B, L), True for padding
            mlm_mask: Optional MLM mask (B, L), True for masked peaks

        Returns:
            Enhanced peak embeddings (B, L, D), or unchanged x if disabled.
        """
        if self.ion_ladder is None:
            return x

        # De-normalize m/z to Daltons
        mz_daltons = spectra[..., 0] * self.max_mz  # (B, L)

        # Compute padding mask
        if spectra_mask is not None:
            pad_mask = spectra_mask.bool()
        else:
            pad_mask = spectra.sum(dim=-1) == 0  # (B, L)

        return self.ion_ladder(mz_daltons, x, pad_mask=pad_mask, mlm_mask=mlm_mask)

    def _apply_padding(self, x: torch.Tensor, spectra_mask: Optional[torch.Tensor], spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply padding replacement for Flash Attention compatibility.

        Args:
            x: Peak embeddings (B, L, D)
            spectra_mask: Optional padding mask (B, L)
            spectra: Original spectra for fallback padding detection (B, L, 2)

        Returns:
            Tuple of (padded_embeddings, padding_mask)
        """
        # Determine padding mask
        if spectra_mask is not None:
            pad_mask = spectra_mask.bool()
        else:
            # Fallback: detect padding from zero spectra
            pad_mask = spectra.sum(dim=-1) == 0

        # Apply pad token replacement (no-op unless Flash Attention is enabled)
        x = self.apply_pad_token_replacement(x, pad_mask)

        return x, pad_mask

    def _apply_mlm_mask(
        self,
        x: torch.Tensor,
        mlm_mask: torch.Tensor,
        spectra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply MLM masking by replacing masked positions with mask token.

        When blur_sigma_da > 0, masked positions receive a Gaussian-blurred
        version of their actual m/z encoding (through the peak encoder) plus
        the learned mask_token as an additive "I am masked" bias.  This gives
        each mask physically meaningful spatial identity while hiding the
        precise m/z value.

        Args:
            x: Peak embeddings (B, L, D)
            mlm_mask: Boolean mask indicating masked positions (B, L)
            spectra: Original spectra (B, L, 2+) for blurred encoding.
                Required when blur_sigma_da > 0.

        Returns:
            Masked embeddings (B, L, D)
        """
        B, L, D = x.shape  # noqa: N806

        # Validate mask shape
        if mlm_mask.shape[1] != L:
            logger.warning(f"MLM mask shape mismatch! Expected {L}, got {mlm_mask.shape[1]}. Fixing by padding/truncating.")
            if mlm_mask.shape[1] < L:
                # Pad mask to match sequence length
                pad_len = L - mlm_mask.shape[1]
                mlm_mask = F.pad(mlm_mask, (0, pad_len), value=False)
            else:
                # Truncate mask
                mlm_mask = mlm_mask[:, :L]

        if self.blur_sigma_da > 0.0 and spectra is None:
            logger.warning("blur_sigma_da > 0 but spectra not provided to _apply_mlm_mask; falling back to learned mask token")

        if self.blur_sigma_da > 0.0 and spectra is not None:
            # Gaussian-blurred mask encoding: noised m/z through peak encoder + mask bias
            sigma_norm = self.blur_sigma_da / self.max_mz
            noise = torch.randn(B, L, 1, device=x.device, dtype=x.dtype) * sigma_norm
            blurred_mz = (spectra[..., 0:1] + noise).clamp(0.0, 1.0)

            # Construct blurred spectra: noised m/z, optionally with true intensity
            blurred_spectra = torch.zeros(B, L, 2, device=x.device, dtype=x.dtype)
            blurred_spectra[..., 0:1] = blurred_mz
            if not self.mask_intensity:
                blurred_spectra[..., 1:2] = spectra[..., 1:2]  # preserve true intensity

            # Encode through peak encoder + add learned mask bias
            blurred_embed = self.peak_encoder(blurred_spectra)
            mask_bias = self.mask_token.expand(B, L, -1)
            x = torch.where(mlm_mask.unsqueeze(-1), blurred_embed + mask_bias, x)
        else:
            # Default: uniform learned mask token
            mask_token_expanded = self.mask_token.expand(B, L, -1)
            x = torch.where(mlm_mask.unsqueeze(-1), mask_token_expanded, x)

        return x

    def _inject_noise(self, spectra: torch.Tensor, mlm_mask: torch.Tensor) -> torch.Tensor:
        """Inject PPM-based m/z noise and proportional intensity noise on masked peaks.

        This follows DreaMS-style noise injection for better robustness.
        Noise is only applied to masked peaks, preserving m/z ordering.

        Args:
            spectra: Input spectra (B, L, 2) [m/z, intensity]
            mlm_mask: Boolean mask indicating masked positions (B, L)

        Returns:
            Noised spectra (B, L, 2)
        """
        if mlm_mask is None or not mlm_mask.any():
            return spectra

        B, L, _ = spectra.shape  # noqa: N806
        device = spectra.device

        # Unpack channels
        mz_norm = spectra[..., 0:1]  # normalized [0,1]
        intensity = spectra[..., 1:2]  # normalized [0,1]
        eff_mask = mlm_mask.unsqueeze(-1)  # (B, L, 1) for broadcasting

        # ===== M/Z Noise (PPM-based) =====
        if self.mz_noise_ppm > 0.0:
            # Convert to Da, apply noise, convert back
            mz_da = mz_norm * self.max_mz
            std_da = mz_da * (self.mz_noise_ppm / 1e6)  # PPM to Da
            std_norm = std_da / self.max_mz  # Back to [0,1]

            # Sample noise only at masked positions
            mz_eps = torch.zeros_like(mz_norm)
            masked_std = std_norm[eff_mask].view(-1)
            if masked_std.numel() > 0:
                mz_eps_masked = torch.normal(mean=0.0, std=masked_std).to(device)
                mz_eps[eff_mask] = mz_eps_masked

            mz_perturbed = (mz_norm + mz_eps).clamp_(0.0, 1.0)

            # Preserve m/z ordering by clamping to neighbor midpoints
            left = torch.roll(mz_norm, shifts=1, dims=1)
            right = torch.roll(mz_norm, shifts=-1, dims=1)
            left[:, 0, :] = 0.0
            right[:, -1, :] = 1.0
            low_bound = 0.5 * (left + mz_norm)
            high_bound = 0.5 * (mz_norm + right)

            # Only enforce ordering where masked
            mz_perturbed = torch.where(
                eff_mask,
                torch.maximum(torch.minimum(mz_perturbed, high_bound), low_bound),
                mz_perturbed,
            )
        else:
            mz_perturbed = mz_norm

        # ===== Intensity Noise (Proportional) =====
        if self.intensity_noise_pct > 0.0:
            intensity_std = (intensity + self.intensity_noise_floor) * self.intensity_noise_pct
            intensity_eps = torch.zeros_like(intensity)
            masked_std = intensity_std[eff_mask].view(-1)
            if masked_std.numel() > 0:
                intensity_eps_masked = torch.normal(mean=0.0, std=masked_std).to(device)
                intensity_eps[eff_mask] = intensity_eps_masked

            intensity_perturbed = (intensity + intensity_eps).clamp_(0.0, 1.0)
        else:
            intensity_perturbed = intensity

        # Reconstruct spectra
        noised = spectra.clone()
        noised[..., 0:1] = mz_perturbed
        noised[..., 1:2] = intensity_perturbed

        return noised

    def _add_special_tokens(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, torch.Tensor]],
        spectra_mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Add latent token and optional meta tokens to sequence.

        Args:
            x: Peak embeddings (B, L, D)
            meta: Optional metadata dictionary
            spectra_mask: Optional spectra padding mask (B, L)
            mlm_mask: Optional MLM mask (B, L)

        Returns:
            Tuple of (updated_x, num_prepended, updated_spectra_mask, updated_mlm_mask)
            where num_prepended includes latent token + all metadata tokens
        """
        B = x.shape[0]  # noqa: N806

        # Add latent token
        x = torch.cat([self.latent_token.expand(B, -1, -1), x], dim=1)
        num_prepended = 1

        # Add meta tokens if enabled and metadata provided
        if self.use_meta_token and meta is not None:
            # Ensure metadata tensors are on correct device
            tensor_meta = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in meta.items()}

            # Get meta token embeddings (B, n_meta_tokens, D)
            meta_tokens = self.meta_embed(tensor_meta)  # (B, n_tokens, D)

            # Insert meta tokens after latent token
            x = torch.cat([x[:, :1], meta_tokens, x[:, 1:]], dim=1)
            num_prepended += meta_tokens.shape[1]  # Add number of meta tokens

            # Pad masks if provided
            if spectra_mask is not None:
                n_meta_tokens = meta_tokens.shape[1]
                spectra_mask = torch.cat([spectra_mask.new_zeros(spectra_mask.size(0), n_meta_tokens), spectra_mask], dim=1)

            if mlm_mask is not None:
                n_meta_tokens = meta_tokens.shape[1]
                mlm_mask = torch.cat([mlm_mask.new_zeros(mlm_mask.size(0), n_meta_tokens), mlm_mask], dim=1)

        return x, num_prepended, spectra_mask, mlm_mask

    def _compute_attn_bias(
        self,
        spectra: torch.Tensor,
        mlm_mask: Optional[torch.Tensor] = None,
        spectra_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute attention bias and pairwise features using PA bias if enabled.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            mlm_mask: Optional MLM mask (B, L), True for masked positions.
                If provided, pairwise features involving masked positions
                are zeroed out to prevent information leakage through pairwise
                m/z distances.
            spectra_mask: Optional padding mask (B, L), True for padded positions.
                If provided, pairwise features involving padded positions
                are zeroed out to prevent meaningless distance computations
                (padded positions have m/z=0 which is not a valid value).

        Returns:
            Tuple of (attn_bias, pairwise_feats):
                - attn_bias: None (PA bias applied per-layer via g_pw)
                - pairwise_feats: (B, L, L, hidden_dim) or None
        """
        if self.pairwise_bias is None:
            return None, None

        # Extract m/z and de-normalize to Daltons.
        # spectra[..., 0] is normalized to [0, 1] by the data processor (÷ max_mz).
        # PairwiseAttentionBias lambda_min/lambda_max are calibrated in Daltons,
        # so passing normalized values makes low-freq components near-flat and
        # high-freq components aliased — destroying the Fourier basis.
        mz = spectra[..., 0:1] * self.max_mz  # (B, L, 1) in Daltons

        # When blur_masked_pa is enabled, substitute blurred m/z at masked
        # positions BEFORE computing PA features.  This gives the encoder
        # approximate (not exact) pairwise m/z information at masked positions
        # instead of zeroing them entirely.
        use_blurred_pa = self.blur_masked_pa and mlm_mask is not None and mlm_mask.any()
        if use_blurred_pa:
            noise = torch.randn_like(mz) * self.blur_sigma_da
            blurred_mz = (mz + noise).clamp(0.0, self.max_mz)
            mz = torch.where(mlm_mask.unsqueeze(-1), blurred_mz, mz)  # type: ignore[union-attr]

        # Compute pairwise features via PairwiseAttentionBias
        pairwise_feats = self.pairwise_bias(mz=mz)  # (B, L, L, hidden_dim)

        # Zero out pairwise features to prevent information leakage.
        # Three modes depending on configuration:
        #   1. blur_masked_pa: blurred m/z at masked positions → only zero padding
        #   2. pa_mask_to_mask: keep exact PA for mask↔mask pairs, zero mask↔visible
        #      cross-pairs (safe — Δm/z between masks reveals relationships, not positions)
        #   3. default: zero ALL pairs involving any masked position
        if use_blurred_pa:
            # Mode 1: only zero padding (masked positions have blurred PA)
            if spectra_mask is not None and spectra_mask.any():
                pad_2d = spectra_mask.unsqueeze(2) | spectra_mask.unsqueeze(1)
                pairwise_feats = pairwise_feats.masked_fill(pad_2d.unsqueeze(-1), 0.0)
        elif self.pa_mask_to_mask and mlm_mask is not None:
            # Mode 2: zero cross-type pairs (mask↔visible) but keep mask↔mask
            cross_mask = mlm_mask.unsqueeze(2) ^ mlm_mask.unsqueeze(1)  # (B, L, L)
            if spectra_mask is not None:
                pad_involved = spectra_mask.unsqueeze(2) | spectra_mask.unsqueeze(1)
                cross_mask = cross_mask | pad_involved
            if cross_mask.any():
                pairwise_feats = pairwise_feats.masked_fill(cross_mask.unsqueeze(-1), 0.0)
        else:
            # Mode 3 (default): zero ALL pairs involving masked or padded positions
            invalid = None
            if mlm_mask is not None:
                invalid = mlm_mask
            if spectra_mask is not None:
                invalid = spectra_mask if invalid is None else (invalid | spectra_mask)
            if invalid is not None and invalid.any():
                invalid_2d = invalid.unsqueeze(2) | invalid.unsqueeze(1)  # (B, L, L)
                pairwise_feats = pairwise_feats.masked_fill(invalid_2d.unsqueeze(-1), 0.0)

        # No attn_bias — the per-layer g_pw projection handles it
        return None, pairwise_feats

    def _pad_attn_bias(
        self,
        attn_bias: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Pad attention bias to account for prepended tokens.

        Args:
            attn_bias: Attention bias (B, H, L, L)
            num_prepended: Number of prepended tokens (latent + meta)

        Returns:
            Padded attention bias (B, H, L+num_prepended, L+num_prepended) or None
        """
        if attn_bias is None:
            return None

        B, H, L, _ = attn_bias.shape  # noqa: N806
        total_len = L + num_prepended

        # Create padded bias with zeros for prepended tokens
        padded = torch.zeros(
            B,
            H,
            total_len,
            total_len,
            device=attn_bias.device,
            dtype=attn_bias.dtype,
        )
        padded[:, :, num_prepended:, num_prepended:] = attn_bias

        return padded

    def _pad_pairwise_feats(
        self,
        pairwise_feats: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Pad pairwise features to account for prepended tokens.

        Args:
            pairwise_feats: Pairwise features (B, L, L, R)
            num_prepended: Number of prepended tokens (latent + meta)

        Returns:
            Padded pairwise features (B, L+num_prepended, L+num_prepended, R) or None
        """
        if pairwise_feats is None or num_prepended == 0:
            return pairwise_feats

        B, L, _, R = pairwise_feats.shape  # noqa: N806
        total_len = L + num_prepended

        padded = torch.zeros(
            B,
            total_len,
            total_len,
            R,
            device=pairwise_feats.device,
            dtype=pairwise_feats.dtype,
        )
        padded[:, num_prepended:, num_prepended:, :] = pairwise_feats

        return padded

    def _create_padding_mask(
        self,
        pad_mask: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Create padding mask for encoder, accounting for prepended tokens.

        Args:
            pad_mask: Original padding mask (B, L)
            num_prepended: Number of prepended tokens

        Returns:
            Padded mask (B, L+num_prepended) or None
        """
        if pad_mask is None:
            return None

        # Prepend zeros for special tokens (not padded)
        prepended_zeros = torch.zeros(
            pad_mask.size(0),
            num_prepended,
            device=pad_mask.device,
            dtype=pad_mask.dtype,
        )
        return torch.cat([prepended_zeros, pad_mask], dim=1)

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: Optional[torch.Tensor] = None,
        spectra_mask: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, torch.Tensor]] = None,
        target_groups: Optional[torch.Tensor] = None,
        target_offsets: Optional[torch.Tensor] = None,
        bin_edges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for masked reconstruction.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            precursors: Optional precursor info (B, 3) [mz, charge, ...] (deprecated, use meta)
            spectra_mask: Optional padding mask (B, L), True for padding
            mlm_mask: Optional MLM mask (B, L), True for masked peaks
            meta: Optional metadata dictionary for meta token
            target_groups: Optional true group indices (B, L) for teacher forcing
                in the offset head during training. None at inference.
            target_offsets: Optional true offset indices (B, L) for teacher forcing
                the delta-ppm head during training. None at inference.
            bin_edges: Optional 1-D bin edges tensor [n_bins+1] in Daltons.
                Required for conditioned delta-ppm during training.

        Returns:
            Tuple of (predictions, auxiliary_outputs) where:
                - predictions: M/z predictions (format depends on mz_task)
                    * regression: (B, L, 1) tensor
                    * classification: tuple of (group_logits, offset_logits)
                - auxiliary_outputs: Dict with auxiliary task predictions
        """
        # 1. Embed peaks
        x = self._embed_peaks(spectra)

        # 1.5 Apply IonLadder (before padding/masking, but with mlm_mask
        # to exclude masked peaks from neighbour windows)
        x = self._apply_ion_ladder(x, spectra, spectra_mask, mlm_mask)

        # 2. Apply padding replacement
        x, pad_mask = self._apply_padding(x, spectra_mask, spectra)

        # 3. Apply MLM masking (if provided)
        if mlm_mask is not None:
            x = self._apply_mlm_mask(x, mlm_mask, spectra=spectra)
            # Inject noise to masked peaks in original spectra
            noised_spectra = self._inject_noise(spectra, mlm_mask)
        else:
            noised_spectra = spectra

        # 4. Compute attention bias and pairwise features (before adding special tokens)
        # Pass mlm_mask and pad_mask to zero out features involving masked/padded
        # positions, preventing information leakage and meaningless distance computations
        attn_bias, pairwise_feats = self._compute_attn_bias(noised_spectra, mlm_mask, pad_mask)

        # 5. Add special tokens (latent + meta)
        x, num_prepended, spectra_mask, mlm_mask = self._add_special_tokens(x, meta, spectra_mask, mlm_mask)

        # 6. Pad attention bias and pairwise features for special tokens
        attn_bias = self._pad_attn_bias(attn_bias, num_prepended)
        pairwise_feats = self._pad_pairwise_feats(pairwise_feats, num_prepended)

        # 7. Create padding mask for encoder.
        # Padding keys MUST be masked (-inf) in every attention softmax on both
        # flash and math backends: learned pad-tokens alone leak an n_peaks axis
        # into the CLS / mean-pool latent (rank-1 contribution proportional to
        # the number of padded slots). FlashMHA accepts key_padding_mask and
        # folds it into the SDPA attn_mask as an additive -inf bias.
        src_key_padding_mask = self._create_padding_mask(pad_mask, num_prepended)

        # 8. Encode through transformer
        if isinstance(self.encoder, nn.TransformerEncoder):
            # Standard PyTorch TransformerEncoder
            x = self.encoder(
                src=x,
                mask=None,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            x = self.encoder(
                src=x,
                src_mask=None,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
                pairwise_feats=pairwise_feats,
                is_causal=False,
            )

        # 9. Extract features and apply prediction heads
        x_tokens = x[:, num_prepended:]  # Remove special tokens
        latent = x[:, 0]  # Extract latent token

        predictions, auxiliary_outputs = self.prediction_heads(
            x_tokens,
            latent,
            target_groups=target_groups,
            target_offsets=target_offsets,
            bin_edges=bin_edges,
        )

        return predictions, auxiliary_outputs

    def forward_with_attn(
        self,
        spectra: torch.Tensor,
        meta: Optional[Dict[str, torch.Tensor]] = None,
        allow_grad: bool = False,
        disable_relative_bias: bool = False,
        return_peak_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass with attention weight capture for visualization.

        This method is used for attention visualization and analysis. It captures
        attention weights from each layer and returns them along with the latent
        embeddings and special token mask.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            meta: Optional metadata dictionary
            allow_grad: Allow gradient computation (default: False)
            disable_relative_bias: Disable relative position bias (default: False)
            return_peak_embeddings: Return peak-level embeddings (default: False)

        Returns:
            Dictionary containing:
                - embeddings: L2-normalized latent embeddings (B, D)
                - attn_weights: List of attention weights per layer [(B, H, T, T), ...]
                - special_mask: Boolean mask for special tokens (B, T)
                    True for latent/meta/padding positions
                - peak_embeddings: Peak-level embeddings (B, L, D) if return_peak_embeddings=True
                    Only includes actual peak positions (excludes latent/meta/padding)
        """
        was_training = self.training
        self.eval()

        try:
            # Context manager for gradient control
            ctx = nullcontext() if allow_grad else torch.no_grad()

            with ctx:
                B, L, _ = spectra.shape  # noqa: N806

                # 1. Embed peaks
                x = self._embed_peaks(spectra)

                # 1.5 Apply IonLadder (before padding/masking)
                x = self._apply_ion_ladder(x, spectra)

                # Snapshot pre-transformer peak embeddings (before CLS/meta prepend and encoder)
                pre_transformer_emb = x.detach().clone()

                # 2. Build padding mask (True = padding)
                pad_mask = spectra.sum(dim=-1) == 0

                # 3. Apply padding replacement
                x = self.apply_pad_token_replacement(x, pad_mask)

                # 4. Compute attention bias and pairwise features (unless disabled)
                # Pass pad_mask to zero out features for padded positions
                if disable_relative_bias:
                    attn_bias = None
                    pairwise_feats = None
                else:
                    attn_bias, pairwise_feats = self._compute_attn_bias(spectra, spectra_mask=pad_mask)

                # 5. Add special tokens
                x = torch.cat([self.latent_token.expand(B, -1, -1), x], dim=1)
                num_prepended = 1

                # Add meta tokens if available
                if self.use_meta_token and meta is not None:
                    tensor_meta = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in meta.items()}

                    meta_tokens = self.meta_embed(tensor_meta)  # (B, n_tokens, D)
                    x = torch.cat([x[:, :1], meta_tokens, x[:, 1:]], dim=1)
                    num_prepended += meta_tokens.shape[1]

                # 6. Create special token mask
                # True for: latent, meta, and padding positions
                special_mask = torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)
                special_mask[:, :num_prepended] = True  # Latent + meta
                special_mask[:, num_prepended:] = pad_mask  # Padding

                # 7. Pad attention bias and pairwise features
                attn_bias = self._pad_attn_bias(attn_bias, num_prepended)
                pairwise_feats = self._pad_pairwise_feats(pairwise_feats, num_prepended)
                if allow_grad and attn_bias is not None:
                    attn_bias = attn_bias.clone()

                # 8. Create key padding mask — always pass it, on both
                # flash and math backends (see padding-mask rule in the
                # Encoder Contract at instanovo_fm/CLAUDE.md §4).
                src_key_padding_mask = None
                if pad_mask is not None:
                    zeros = torch.zeros(B, num_prepended, dtype=pad_mask.dtype, device=pad_mask.device)
                    src_key_padding_mask = torch.cat([zeros, pad_mask], dim=1)

                # 9. Clone pairwise features if gradient tracking needed
                if allow_grad and pairwise_feats is not None:
                    pairwise_feats = pairwise_feats.clone()

                # 10. Capture attention weights with forward hooks
                attn_weights: list[Any] = []
                hooks = []

                if hasattr(self.encoder, "layers"):
                    layers_list = list(self.encoder.layers)  # Convert to list for enumerate
                    for layer_idx, layer in enumerate(layers_list):
                        if hasattr(layer, "self_attn"):
                            # Use closure to capture layer index and ensure correct ordering
                            def _make_hook(layer_index: Any) -> Any:
                                def _hook(module: Any, inputs: Any, output: Any) -> None:
                                    # Many attention modules return (output, attn_weights)
                                    if isinstance(output, tuple) and len(output) == 2:
                                        _, aw = output
                                    else:
                                        aw = None

                                    # Clone to break inference tensor provenance
                                    if allow_grad and aw is not None:
                                        aw = aw.clone()

                                    # Ensure we have the right number of layers
                                    while len(attn_weights) <= layer_index:
                                        attn_weights.append(None)
                                    attn_weights[layer_index] = aw

                                return _hook

                            hooks.append(layer.self_attn.register_forward_hook(_make_hook(layer_idx)))

                # 11. Forward through encoder
                if isinstance(self.encoder, nn.TransformerEncoder):
                    x = self.encoder(
                        src=x,
                        mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                else:
                    # Pass return_attn_weights=True to force attention weight computation
                    x = self.encoder(
                        src=x,
                        src_mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                        attn_bias=attn_bias,
                        pairwise_feats=pairwise_feats,
                        is_causal=False,
                        return_attn_weights=True,  # Force attention weight computation for visualization
                    )

                # 12. Remove hooks
                for hook in hooks:
                    hook.remove()

                # 13. Extract and normalize latent embedding
                latent = x[:, 0]
                latent_normalized = F.normalize(latent, p=2, dim=-1)

                result: dict[str, Any] = {
                    "embeddings": latent_normalized,
                    "attn_weights": attn_weights if attn_weights else [None] * self.n_layers,
                    "special_mask": special_mask,
                }

                # 14. Optionally extract peak embeddings
                if return_peak_embeddings:
                    # Extract peak embeddings (exclude latent/meta tokens, keep padding for masking)
                    peak_emb = x[:, num_prepended:]  # (B, L, D)
                    result["peak_embeddings"] = peak_emb
                    result["peak_embeddings_pretransformer"] = pre_transformer_emb
                    result["num_prepended"] = num_prepended

                return result

        finally:
            if was_training:
                self.train()

    def encode(
        self,
        spectra: torch.Tensor,
        meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extract L2-normalized latent embeddings for inference.

        This method is used for extracting spectral embeddings without computing
        predictions. Useful for downstream tasks like similarity search, clustering,
        or transfer learning.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            meta: Optional metadata dictionary

        Returns:
            L2-normalized latent embeddings (B, D)
        """
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                B, L, _ = spectra.shape  # noqa: N806

                # 1. Embed peaks
                x = self._embed_peaks(spectra)

                # 1.5 Apply IonLadder (before padding)
                x = self._apply_ion_ladder(x, spectra)

                # 2. Apply padding
                pad_mask = spectra.sum(dim=-1) == 0
                x = self.apply_pad_token_replacement(x, pad_mask)

                # 3. Compute attention bias and pairwise features
                attn_bias, pairwise_feats = self._compute_attn_bias(spectra, spectra_mask=pad_mask)

                # 4. Add special tokens
                x, num_prepended, _, _ = self._add_special_tokens(x, meta, None, None)

                # 5. Pad attention bias and pairwise features
                attn_bias = self._pad_attn_bias(attn_bias, num_prepended)
                pairwise_feats = self._pad_pairwise_feats(pairwise_feats, num_prepended)

                # 6. Create padding mask — always required, on both flash and
                # math backends, to prevent the learned pad-token from leaking
                # into the latent via unmasked softmax over padded keys.
                src_key_padding_mask = self._create_padding_mask(pad_mask, num_prepended)

                # 7. Encode
                if isinstance(self.encoder, nn.TransformerEncoder):
                    x = self.encoder(
                        src=x,
                        mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                else:
                    x = self.encoder(
                        src=x,
                        src_mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                        attn_bias=attn_bias,
                        pairwise_feats=pairwise_feats,
                        is_causal=False,
                    )

                # 8. Extract and normalize latent
                latent = x[:, 0]
                latent_normalized = F.normalize(latent, p=2, dim=-1)

                return latent_normalized.cpu()

        finally:
            if was_training:
                self.train()

    def encode_mean_pooled(
        self,
        spectra: torch.Tensor,
        meta: Optional[Dict[str, torch.Tensor]] = None,
        pooling: str = "mean_pool",
        confidence_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Extract L2-normalized pooled peak token embeddings.

        Pools over non-padding peak tokens (excludes CLS and meta tokens).

        Pooling modes:
            - "mean_pool": Uniform mean over all non-padding peaks.
            - "confidence": Weight each peak by the model's classification
              confidence (softmax probability). High-confidence peaks (signal)
              contribute more than low-confidence peaks (noise). Requires a
              classification model; falls back to mean_pool otherwise.

        Args:
            spectra: Input spectra (B, L, 2 or 3)
            meta: Optional metadata dictionary
            pooling: Pooling strategy ("mean_pool" or "confidence")
            confidence_temperature: Softmax temperature for confidence weights
                (only used when pooling="confidence"). T<1 sharpens, T>1 smooths.

        Returns:
            L2-normalized pooled embeddings (B, D)
        """
        use_confidence = pooling == "confidence"
        if use_confidence and self.mz_task != "classification":
            logger.warning("Confidence pooling requires classification model. Falling back to uniform mean pooling.")
            use_confidence = False

        if use_confidence:
            from instanovo_fm.trainer.losses import compute_classifier_confidence

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                B, L, _ = spectra.shape  # noqa: N806

                # 1. Embed peaks
                x = self._embed_peaks(spectra)

                # 1.5 Apply IonLadder (before padding)
                x = self._apply_ion_ladder(x, spectra)

                # 2. Apply padding
                pad_mask = spectra.sum(dim=-1) == 0
                x = self.apply_pad_token_replacement(x, pad_mask)

                # 3. Compute attention bias and pairwise features
                attn_bias, pairwise_feats = self._compute_attn_bias(spectra, spectra_mask=pad_mask)

                # 4. Add special tokens
                x, num_prepended, _, _ = self._add_special_tokens(x, meta, None, None)

                # 5. Pad attention bias and pairwise features
                attn_bias = self._pad_attn_bias(attn_bias, num_prepended)
                pairwise_feats = self._pad_pairwise_feats(pairwise_feats, num_prepended)

                # 6. Create padding mask — always required, on both flash and
                # math backends, to prevent the learned pad-token from leaking
                # into the latent via unmasked softmax over padded keys.
                src_key_padding_mask = self._create_padding_mask(pad_mask, num_prepended)

                # 7. Encode
                if isinstance(self.encoder, nn.TransformerEncoder):
                    x = self.encoder(
                        src=x,
                        mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                else:
                    x = self.encoder(
                        src=x,
                        src_mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                        attn_bias=attn_bias,
                        pairwise_feats=pairwise_feats,
                        is_causal=False,
                    )

                # 8. Extract peak tokens (exclude CLS + meta tokens)
                peak_tokens = x[:, num_prepended:]  # (B, L, D)
                valid_mask = ~pad_mask  # True where real peaks exist

                # 9. Compute pooling weights
                if use_confidence:
                    latent = x[:, 0]
                    preds, _ = self.prediction_heads(peak_tokens, latent)
                    group_logits, offset_logits = preds

                    n_groups = int(self.n_bin_groups_tensor.item())
                    last_group_size = int(self.last_group_size_tensor.item())
                    conf_dict = compute_classifier_confidence(
                        group_logits,
                        offset_logits,
                        n_groups,
                        last_group_size,
                    )
                    conf = conf_dict["conf_joint"]  # (B, L)

                    conf = conf.masked_fill(pad_mask, -float("inf"))
                    weights = torch.softmax(conf / confidence_temperature, dim=1)  # (B, L)

                    # Handle all-padding edge case (softmax over all -inf → NaN)
                    all_padded = ~valid_mask.any(dim=1, keepdim=True)  # (B, 1)
                    if all_padded.any():
                        weights = weights.masked_fill(all_padded, 0.0)

                    pooled = (peak_tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
                else:
                    # Uniform mean pooling
                    valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L, 1)
                    sum_embeddings = (peak_tokens * valid_mask_expanded).sum(dim=1)  # (B, D)
                    count = valid_mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
                    pooled = sum_embeddings / count  # (B, D)

                return F.normalize(pooled, p=2, dim=-1).cpu()

        finally:
            if was_training:
                self.train()

    def clear_rope_cache(self) -> None:
        """Clear RoPE position encoding caches in all attention layers.

        This should be called before gradient-based attribution methods (e.g.,
        Captum Integrated Gradients) if the model was previously used under
        torch.no_grad() or torch.inference_mode(). The RoPE cache stores
        cos/sin tensors that become "inference tensors" under these contexts,
        which cannot be saved for backward passes.

        After clearing, the next forward pass will regenerate the cache with
        properly detached tensors that work with autograd.
        """
        for module in self.modules():
            if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
                if hasattr(module.rotary_emb, "clear_cache"):
                    module.rotary_emb.clear_cache()
            if hasattr(module, "clear_cache"):
                # Direct RoPE modules
                if module.__class__.__name__ == "SimpleRotaryEmbedding":
                    module.clear_cache()

    @staticmethod
    def get_pretrained() -> list[str]:
        """Get a list of pretrained foundation model IDs.

        Returns:
            List of available pretrained model IDs
        """
        try:
            with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
                models_config = json.load(f)

            if MODEL_TYPE not in models_config:
                return []

            return list(models_config[MODEL_TYPE].keys())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @classmethod
    def load(cls, path: str) -> Tuple["FoundationModel", DictConfig]:
        """Load model from checkpoint path.

        Handles two checkpoint generations:
        - Older (pre-torch.compile): has bin_size_tensor, flat binning config,
          no bin_edges.
        - Newer (torch.compile'd): _orig_mod. prefix, bin_edges, nested
          binning config.

        Args:
            path: Path to checkpoint file

        Returns:
            Tuple of (model, config)
        """
        from omegaconf import OmegaConf

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing 'state_dict' key: {path}")

        config = ckpt["config"]
        state_dict = ckpt["state_dict"]

        # ── Step 1: Strip _orig_mod. prefix (MUST be first) ──
        orig_mod_keys = [k for k in state_dict if k.startswith("_orig_mod.")]
        if orig_mod_keys:
            logger.info(f"Stripping '_orig_mod.' prefix from {len(orig_mod_keys)} torch.compile'd checkpoint keys")
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

        # ── Step 2: Remove legacy state_dict keys ──
        if "bin_size_tensor" in state_dict:
            del state_dict["bin_size_tensor"]
            logger.info("Removed legacy 'bin_size_tensor' from checkpoint state_dict")

        # ── Step 3: Migrate old config format ──
        mz_head_cfg = config.get("mz_head", {})
        if "binning" not in mz_head_cfg:
            config["mz_head"]["binning"] = {
                "strategy": "fixed_da",
                "bin_size": mz_head_cfg.get("bin_size", 0.02),
            }
            logger.debug("Migrated old checkpoint to new binning format (fixed_da)")

        if "bin_edges" in state_dict:
            strategy_name = config["mz_head"]["binning"].get("strategy", "fixed_da")
            logger.debug(f"Loading checkpoint with {strategy_name} binning strategy")

        # ── Step 4: Extract model parameters (always nested format) ──
        mz_task = config.get("mz_head", {}).get("task", "regression")
        peak_encoder_type = config.get("peak_encoder", {}).get("type", "multiscale")
        use_meta_token = config.get("meta_token", {}).get("enabled", True)
        min_mz = config.get("min_mz", 0.0)
        max_mz = config.get("max_mz", 2500.0)

        # ── Step 5: Architecture verification ──
        # Old checkpoints used n_bins = int((max_mz - min_mz) / bin_size) + 1,
        # which produces one extra group. Adjust max_mz for fixed_da to match.
        if mz_task == "classification":
            from instanovo_fm.trainer.binning import create_binning_strategy

            group_head_key = "prediction_heads.mz_head.group_head.2.weight"
            if group_head_key in state_dict:
                n_groups_ckpt = state_dict[group_head_key].shape[0]
                strategy = create_binning_strategy(config, min_mz, max_mz)
                n_groups_current = strategy.n_groups

                if n_groups_ckpt != n_groups_current:
                    binning_cfg = config["mz_head"]["binning"]
                    strategy_name = binning_cfg.get("strategy", "fixed_da")

                    if strategy_name == "fixed_da":
                        bin_size = binning_cfg.get("bin_size", 0.02)
                        max_mz_adjusted = max_mz + bin_size
                        strategy_adj = create_binning_strategy(config, min_mz, max_mz_adjusted)
                        if strategy_adj.n_groups == n_groups_ckpt:
                            logger.info(
                                f"Adjusting max_mz from {max_mz} to {max_mz_adjusted} Da to match old checkpoint formula ({n_groups_ckpt} groups)"
                            )
                            max_mz = max_mz_adjusted
                        else:
                            logger.warning(
                                f"Architecture mismatch: checkpoint has "
                                f"{n_groups_ckpt} classification groups but "
                                f"current config produces {n_groups_current}."
                            )
                    else:
                        logger.warning(
                            f"Architecture mismatch: checkpoint has "
                            f"{n_groups_ckpt} classification groups but "
                            f"current config produces {n_groups_current}. "
                            f"Check min_mz, max_mz, and binning config."
                        )

        # ── Step 5b: Inject residue masses for ion_ladder if missing ──
        ion_ladder_cfg = config.get("ion_ladder", {})
        if ion_ladder_cfg.get("enabled", False) and not ion_ladder_cfg.get("residue_masses"):
            if "residues" in ckpt:
                config.setdefault("ion_ladder", {})["residue_masses"] = ckpt["residues"]
                logger.info(f"Injected {len(ckpt['residues'])} residue masses from checkpoint into ion_ladder config")
            else:
                logger.warning("ion_ladder.enabled=True but no residue_masses in config or checkpoint")

        # ── Step 6: Construct model ──
        model = cls(
            dim_model=config.get("dim_model", 512),
            n_heads=config.get("n_heads", 8),
            dim_feedforward=config.get("dim_feedforward", 2048),
            n_layers=config.get("n_layers", 6),
            dropout=config.get("dropout", 0.1),
            n_peaks=config.get("n_peaks", 200),
            max_mz=max_mz,
            min_mz=min_mz,
            max_charge=config.get("max_charge", 10),
            peak_encoder_type=peak_encoder_type,
            mz_task=mz_task,
            use_meta_token=use_meta_token,
            cfg=config,
        )

        # ── Step 7: Load weights ──
        incompatible_keys = model.load_state_dict(state_dict, strict=False)

        expected_missing = {"bin_edges"}
        if incompatible_keys.missing_keys:
            unexpected_missing = [
                k
                for k in incompatible_keys.missing_keys
                if k not in expected_missing
                and not k.startswith("prediction_heads.mz_head.group_embedding.")
                # Batched PA projection is new — old checkpoints won't have it
                and not k.startswith("encoder.pw_projection.")
                # IonLadder and DualPeakEmbedding are new — old checkpoints won't have them
                and not k.startswith("ion_ladder.")
                and not k.startswith("peak_encoder.rbf_")
            ]
            if unexpected_missing:
                logger.warning(f"Missing keys in checkpoint: {unexpected_missing}")

        if incompatible_keys.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}")

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        return model, config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
    ) -> Tuple["FoundationModel", DictConfig]:
        """Download and load pretrained model by ID or path.

        Args:
            model_id: Model ID from models.json or local file path

        Returns:
            Tuple of (model, config)
        """
        # Check if model_id is a local file path
        if "/" in model_id or "\\" in model_id or model_id.endswith(".ckpt"):
            if os.path.isfile(model_id):
                return cls.load(model_id)
            else:
                raise FileNotFoundError(f"No file found at path: {model_id}")

        # Load models.json
        with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        # Find model in config
        if MODEL_TYPE not in models_config or model_id not in models_config[MODEL_TYPE]:
            available = list(models_config.get(MODEL_TYPE, {}).keys())
            raise ValueError(f"Model {model_id} not found in models.json. Available {MODEL_TYPE} models: {available}")

        model_info = models_config[MODEL_TYPE][model_id]
        url = model_info["remote"]

        # Create cache directory
        cache_dir = Path.home() / ".cache" / "instanovo"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        file_name = urlsplit(url).path.split("/")[-1]
        cached_file = cache_dir / file_name

        # Download if not cached
        if not cached_file.exists():
            download_file(url, cached_file, model_id, file_name)
        else:
            logger.info(f"Model {model_id} already cached at {cached_file}")

        try:
            # Load and return
            logger.info(f"Loading model {model_id} (remote)")
            return cls.load(str(cached_file))
        except Exception as e:
            logger.warning(f"Failed to load cached model {model_id}, may be corrupted. Deleting and re-downloading. Error: {e}")
            if cached_file.exists():
                cached_file.unlink()

            download_file(url, cached_file, model_id, file_name)
            logger.info(f"Loading newly downloaded model {model_id}")
            return cls.load(str(cached_file))
