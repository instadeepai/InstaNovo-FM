"""Ion-ladder encoder for charge-aware residue mass matching.

For each peak in a spectrum, the encoder computes mass differences to its
*k* nearest neighbours (by m/z), then soft-matches the absolute differences
against known residue masses under multiple charge hypotheses using Gaussian
kernels.  The resulting per-peak feature vector indicates which amino-acid
residues may span between neighbouring fragment ions.  A zero-initialised
gate (ReZero; Bachlechner et al. 2020, arXiv:2003.04887) is used so the
model starts conservatively and learns how much to incorporate the
ion-ladder signal.

Inspired by ion ladder theory in tandem mass spectrometry and pairwise
mass-difference features from Lapin et al. 2025 ("Pairwise Attention:
Leveraging Mass Differences to Enhance De Novo Sequencing of Mass
Spectra", J. Proteome Res. 24(7):3722--3730,
doi:10.1021/acs.jproteome.5c00063).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IonLadderEncoder(nn.Module):
    """Charge-aware ion-ladder encoding for peak features.

    For each peak, computes mass differences to *k* nearest neighbours
    (by m/z), then soft-matches against residue masses under charge
    hypotheses (default z = 1, 2).  Produces per-peak features that are
    projected to model dimension and added to the peak embeddings via a
    learned, zero-initialised gate (ReZero).

    When ``neutral_losses`` is provided, the reference mass table is expanded
    with loss-shifted variants (e.g. residue_mass − 18.011 for H₂O loss).
    This captures ion-ladder connections where one fragment has undergone a
    neutral loss — common for Ser/Thr (H₂O) and Asn/Gln/Lys/Arg (NH₃).

    Args:
        d_model: Model / peak embedding dimension.
        residue_masses: List of positive residue masses (Da).
        window_k: Number of nearest neighbours on each side.
        sigma: Standard deviation of the Gaussian soft-matching kernel.
        charge_states: Charge hypotheses for mass-difference scaling.
        neutral_losses: Optional mapping of loss name → mass in Da
            (e.g. ``{"H2O": 18.010565, "NH3": 17.026549}``).  When given,
            each residue mass is expanded with ``residue ± loss`` variants,
            tripling (or more) the soft-match feature dimension.
    """

    # Standard neutral loss masses (Da)
    KNOWN_LOSSES: dict[str, float] = {
        "H2O": 18.010565,
        "NH3": 17.026549,
    }

    def __init__(
        self,
        d_model: int,
        residue_masses: list[float],
        window_k: int = 40,
        sigma: float = 0.35,
        charge_states: tuple[int, ...] = (1, 2),
        neutral_losses: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.window_k = window_k

        self.register_buffer(
            "charges", torch.tensor(charge_states, dtype=torch.float32)
        )
        self.register_buffer(
            "inv_sigma_sq", torch.tensor(1.0 / (sigma**2))
        )

        # Build expanded reference mass table: bare masses + loss-shifted variants
        ref_masses = sorted(residue_masses)  # bare residue masses
        if neutral_losses:
            for loss_mass in neutral_losses.values():
                for rm in sorted(residue_masses):
                    shifted = rm + loss_mass
                    if shifted > 0:
                        ref_masses.append(shifted)

        self.register_buffer(
            "residue_masses_buf",
            torch.tensor(sorted(ref_masses), dtype=torch.float32),
        )

        n_ref_masses = len(ref_masses)
        n_charges = len(charge_states)
        feature_dim = n_ref_masses * n_charges

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
        )
        self.gate_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        mz_daltons: Tensor,
        peak_embeddings: Tensor,
        pad_mask: Tensor | None = None,
        mlm_mask: Tensor | None = None,
    ) -> Tensor:
        """Enhance peak embeddings with ion-ladder features.

        Args:
            mz_daltons: (B, L) m/z values in Daltons (NOT normalized).
            peak_embeddings: (B, L, D) from peak encoder.
            pad_mask: (B, L) bool -- True for *padding* (invalid) peaks.
            mlm_mask: (B, L) bool -- True for *masked* peaks (MLM).
                Masked peaks are excluded from neighbour windows to prevent
                information leakage: an unmasked peak must not see the m/z
                position of a masked neighbour, as this would reveal the
                reconstruction target through residue-mass matching.

        Returns:
            (B, L, D) enhanced peak embeddings.
        """
        k = self.window_k
        mz = mz_daltons  # (B, L)

        # Combine padding and MLM masks into a single "invalid" mask.
        # Invalid peaks have their m/z zeroed and are excluded from
        # neighbour windows to prevent information leakage.
        invalid_mask = torch.zeros_like(mz, dtype=torch.bool)
        if pad_mask is not None:
            invalid_mask = invalid_mask | pad_mask
        if mlm_mask is not None:
            invalid_mask = invalid_mask | mlm_mask

        # Zero out invalid m/z values
        if invalid_mask.any():
            mz = mz.masked_fill(invalid_mask, 0.0)

        # Pad m/z with zeros on each side for windowing
        mz_padded = F.pad(mz, (k, k), value=0.0)  # (B, L+2k)

        # Pad the invalid mask (boundary positions are treated as invalid)
        invalid_mask_padded = F.pad(invalid_mask, (k, k), value=True)

        # Unfold to get neighbour windows
        windows = mz_padded.unfold(1, 2 * k + 1, 1)  # (B, L, 2k+1)
        mask_windows = invalid_mask_padded.unfold(1, 2 * k + 1, 1)  # (B, L, 2k+1)

        # Compute mass diffs (exclude self at center position k)
        center = mz.unsqueeze(-1)  # (B, L, 1)
        diffs = windows - center  # (B, L, 2k+1)

        # Remove center (self) position
        neighbor_diffs = torch.cat(
            [diffs[:, :, :k], diffs[:, :, k + 1 :]], dim=2
        )  # (B, L, 2k)
        neighbor_mask = torch.cat(
            [mask_windows[:, :, :k], mask_windows[:, :, k + 1 :]], dim=2
        )  # (B, L, 2k)

        # Valid neighbour mask for zeroing out invalid contributions
        neighbor_valid = (~neighbor_mask).unsqueeze(-1).float()  # (B, L, 2k, 1)

        abs_diffs = neighbor_diffs.abs()  # (B, L, 2k)

        # Soft-match against residue masses under each charge hypothesis
        features_per_charge: list[Tensor] = []
        for c_idx in range(len(self.charges)):
            z = self.charges[c_idx]
            scaled = abs_diffs * z  # (B, L, 2k)
            # Gaussian kernel: exp(-(scaled - mass)^2 / sigma^2)
            delta = scaled.unsqueeze(-1) - self.residue_masses_buf  # (B, L, 2k, R)
            soft_match = torch.exp(-delta.pow(2) * self.inv_sigma_sq)
            soft_match = soft_match * neighbor_valid
            # Sum over neighbours → (B, L, R)
            features_per_charge.append(soft_match.sum(dim=2))

        # Concat charges → (B, L, n_charges * n_residues)
        ladder_feats = torch.cat(features_per_charge, dim=-1)

        # Project to model dim and apply ReZero gate
        projected = self.proj(ladder_feats)  # (B, L, D)
        gate = torch.sigmoid(self.gate_scale)

        return peak_embeddings + gate * projected
