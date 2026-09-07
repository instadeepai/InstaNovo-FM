"""Data processor for InstaNovo Foundation Model.

This module provides FoundationalDataProcessor for self-supervised learning
on mass spectrometry data through masked m/z reconstruction.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np
import spectrum_utils.spectrum as sus
import torch
from jaxtyping import Float
from torch import Tensor, nn

from instanovo.__init__ import console
from instanovo.common import DataProcessor
from instanovo.constants import ANNOTATED_COLUMN, PROTON_MASS_AMU, MSColumns
from instanovo_fm.data.masking import get_mask_function
from instanovo_fm.data.search_data_manager import SearchDataManager
from instanovo.utils.residues import ResidueSet
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Type aliases for configuration options
MaskingStrategy = Literal["thompson", "thompson_span", "signal_aware_span", "uniform", "none"]
PeakOrdering = Literal["sorted", "cyclic_shift", "complete_shuffle"]


class FoundationalDataProcessor(DataProcessor):
    """Data processor for self-supervised learning on MS/MS spectra.
    
    Designed for encoder-only foundation model that learns representations through
    masked m/z reconstruction. Does not require peptide annotations for training.
    
    Processing Pipeline:
        1. Spectrum preprocessing (filter, normalize, scale)
        2. Fixed-length padding to n_peaks
        3. Optional peak reordering (tests positional encoding robustness)
        4. Masking strategy application (self-supervised learning objective)
    
    Key Features:
        - Fixed-length padding (consistent tensor shapes)
        - Charge-aware isotope detection
        - Multiple masking strategies
        - Optional sequence processing (for evaluation/analysis only)
    
    Args:
        n_peaks: Fixed number of peaks per spectrum (default: 200).
        min_mz: Minimum m/z value to retain (default: 50.0).
        max_mz: Maximum m/z value to retain (default: 2500.0).
        min_intensity: Minimum intensity threshold (default: 0.01).
        remove_precursor_tol: Precursor removal tolerance in Da (default: 2.0).
        use_spectrum_utils: Use spectrum_utils library for preprocessing (default: True).
        normalize_mz: Normalize m/z to [0,1] by dividing by max_mz (default: True).
        peak_ordering: Peak ordering strategy (default: "sorted").
            - "sorted": Keep m/z order (preserves spectral patterns)
            - "cyclic_shift": Random rotation (data augmentation)
            - "complete_shuffle": Full randomization (hardest test)
        residue_set: ResidueSet for sequence tokenization (optional, for evaluation).
        annotated: Include sequence processing for evaluation (default: False).
        return_str: Keep sequences as strings vs tokenized (default: True).
        metadata_columns: Additional columns to include in batches.
        masking_strategy: Strategy for masked learning (default: "thompson_span").
        mask_portion: Fraction of peaks to mask (default: 0.30).
        thompson_alpha: Thompson sampling Beta prior alpha (default: 0.5).
        thompson_beta: Thompson sampling Beta prior beta (default: 0.5).
        thompson_kappa: Intensity weighting concentration (default: 4.0).
        thompson_gamma: Intensity tempering exponent (default: 0.7).
        span_min: Minimum span length for span masking (default: 4).
        span_max: Maximum span length for span masking (default: 7).
        span_bidirectional: Expand spans bidirectionally (default: True).
        include_isotopes: Co-mask isotopic neighbors (default: True).
        isotope_ppm: PPM tolerance for isotope detection (default: 25.0).
        isotope_da_floor: Minimum Da tolerance for isotopes (default: 0.02).
        isotope_max_charge: Fallback max charge if not known (default: 3).
        isotope_max_order: Maximum isotopic order M+n (default: 2).
        max_total_mask_ratio: Hard cap on total masking ratio (default: 0.40).
            Never exceed this percentage, even with isotope co-masking.
        search_data_manager: Optional SearchDataManager for additional metadata (default: None).
    """

    def __init__(
        self,
        # Spectrum processing
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 0.0,
        use_spectrum_utils: bool = True,
        normalize_mz: bool = True,
        peak_ordering: PeakOrdering = "sorted",
        # Sequence processing (optional, for evaluation)
        residue_set: ResidueSet | None = None,
        annotated: bool = False,
        return_str: bool = True,
        metadata_columns: list[str] | None = None,
        # Masking configuration
        masking_strategy: MaskingStrategy = "thompson_span",
        mask_portion: float = 0.30,
        # Thompson sampling
        thompson_alpha: float = 0.5,
        thompson_beta: float = 0.5,
        thompson_kappa: float = 4.0,
        thompson_gamma: float = 0.7,
        # Span masking
        span_min: int = 4,
        span_max: int = 7,
        span_bidirectional: bool = True,
        # Isotope co-masking
        include_isotopes: bool = True,
        isotope_ppm: float = 25.0,
        isotope_da_floor: float = 0.02,
        isotope_max_charge: int = 3,
        isotope_max_order: int = 2,
        max_total_mask_ratio: float = 0.40,  # Hard cap on total masking
        # Signal-aware masking (requires sequences)
        signal_min_backbone_coverage: float = 0.15,
        signal_min_fragment_groups: int = 3,
        signal_ppm: float = 20.0,
        signal_cid_da_tol: float = 0.2,
        signal_ion_types: tuple[str, ...] = ("b", "y"),
        signal_num_workers: int = 4,
        # Search data integration (optional)
        search_data_manager: SearchDataManager | None = None,
        # Meta token configuration (optional)
        build_metadata: bool = False,
    ) -> None:
        """Initialize the foundation model data processor."""
        super().__init__(metadata_columns=metadata_columns)

        # Track missing metadata columns to warn once (not per-row)
        self._warned_missing_columns: set[str] = set()

        # Store spectrum processing config
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.use_spectrum_utils = use_spectrum_utils
        self.normalize_mz = normalize_mz
        self.peak_ordering = peak_ordering
        
        # Store sequence processing config (optional)
        self.residue_set = residue_set
        self.annotated = annotated
        self.return_str = return_str
        
        # Store masking config
        self.masking_strategy = masking_strategy
        self.mask_portion = mask_portion
        self.thompson_alpha = thompson_alpha
        self.thompson_beta = thompson_beta
        self.thompson_kappa = thompson_kappa
        self.thompson_gamma = thompson_gamma
        self.span_min = span_min
        self.span_max = span_max
        self.span_bidirectional = span_bidirectional
        self.include_isotopes = include_isotopes
        self.isotope_ppm = isotope_ppm
        self.isotope_da_floor = isotope_da_floor
        self.isotope_max_charge = isotope_max_charge
        self.isotope_max_order = isotope_max_order
        self.max_total_mask_ratio = max_total_mask_ratio

        # Store signal-aware masking config
        self.signal_min_backbone_coverage = signal_min_backbone_coverage
        self.signal_min_fragment_groups = signal_min_fragment_groups
        self.signal_ppm = signal_ppm
        self.signal_cid_da_tol = signal_cid_da_tol
        self.signal_ion_types = signal_ion_types
        self.signal_num_workers = signal_num_workers

        # Store search data manager (optional)
        self.search_data_manager = search_data_manager
        
        # Store metadata building flag
        self.build_metadata = build_metadata
        
        # Validate: tokenization requires residue_set
        if self.annotated and self.residue_set is None and not self.return_str:
            raise ValueError(
                "residue_set required for sequence tokenization. "
                "Set return_str=True to keep sequences as strings."
            )

        # Log initialization
        mode = "eval" if self.annotated else "train"
        logger.debug(
            f"DataProcessor: n_peaks={n_peaks}, mode={mode}, masking={masking_strategy}"
        )

    def _process_spectrum(
        self,
        mz_array: Float[Tensor, " peak"],
        int_array: Float[Tensor, " peak"],
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """Process single spectrum: filter, normalize, scale.
        
        Pipeline:
            1. Filter m/z range [min_mz, max_mz]
            2. Remove precursor peak (±tolerance)
            3. Filter by intensity threshold
            4. Keep top n_peaks by intensity
            5. Apply √intensity scaling
            6. L2-normalize intensities
            7. Normalize m/z to [0,1] (optional)
        
        Args:
            mz_array: Peak m/z values.
            int_array: Peak intensities.
            precursor_mz: Precursor m/z value.
            precursor_charge: Precursor charge state.
            
        Returns:
            Processed spectrum tensor [n_peaks, 2] with [m/z, intensity] pairs.
            Returns dummy spectrum [[0, 1]] if processing fails.
        """
        # Use spectrum_utils library if available (faster)
        if self.use_spectrum_utils:
            spectrum = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                np.array(mz_array).astype(np.float32),
                np.array(int_array).astype(np.float32),
            )
            try:
                # Apply filters
                spectrum.set_mz_range(self.min_mz, self.max_mz)
                if len(spectrum.mz) == 0:
                    raise ValueError("Empty after m/z filtering")
                
                spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
                if len(spectrum.mz) == 0:
                    raise ValueError("Empty after precursor removal")
                
                spectrum.filter_intensity(self.min_intensity, self.n_peaks)
                if len(spectrum.mz) == 0:
                    raise ValueError("Empty after intensity filtering")
                
                # Scale and normalize intensities
                # Why √ transformation for proteomic MS/MS?
                # - MS/MS produces discrete ion counts (Poisson-like noise)
                # - Variance proportional to mean: Var(X) = λ
                # - √ transformation stabilizes variance: Var(√X) ≈ constant
                # - Creates homoscedastic errors (better for regression)
                # Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC1343595/
                spectrum.scale_intensity("root", 1)  # √intensity

                # Why L2-normalization?
                # - Normalizes for total ion current (TIC) variations
                # - TIC varies across acquisitions due to:
                #   * Sample concentration differences
                #   * Ionization efficiency variations
                #   * Detector sensitivity drift
                # - Creates unit-norm vector: ||intensity||₂ = 1
                # - Preserves relative intensity patterns (critical for fragment identification)
                # - Makes spectra comparable across different acquisition conditions
                # - Output: Unit-norm vector (individual values typically in [0, ~0.5])
                intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)  # L2-norm
                
                # Optionally normalize m/z to [0,1]
                mz_values = spectrum.mz / self.max_mz if self.normalize_mz else spectrum.mz
                
                return torch.tensor(np.array([mz_values, intensities])).T.float()
                
            except ValueError as e:
                logger.debug(f"Spectrum processing failed: {e}")
                # Return dummy spectrum
                dummy_mz = 0.0 if self.normalize_mz else 0.0
                return torch.tensor([[dummy_mz, 1.0]], dtype=torch.float32)

        # PyTorch fallback implementation (if spectrum_utils unavailable)
        try:
            # Step 1: Filter m/z range
            mask = (mz_array >= self.min_mz) & (mz_array <= self.max_mz)
            mz_array, int_array = mz_array[mask], int_array[mask]
            if len(mz_array) == 0:
                raise ValueError("Empty after m/z filtering")

            # Step 2: Remove precursor
            mask = torch.abs(mz_array - precursor_mz) > self.remove_precursor_tol
            mz_array, int_array = mz_array[mask], int_array[mask]
            if len(mz_array) == 0:
                raise ValueError("Empty after precursor removal")

            # Step 3: Filter by intensity
            mask = int_array >= self.min_intensity
            mz_array, int_array = mz_array[mask], int_array[mask]
            if len(mz_array) == 0:
                raise ValueError("Empty after intensity filtering")

            # Step 4: Keep top n_peaks
            if len(mz_array) > self.n_peaks:
                _, indices = torch.topk(int_array, self.n_peaks)
                mz_array, int_array = mz_array[indices], int_array[indices]

            # Step 5: Scale intensity (√)
            # Variance stabilization for Poisson-like ion count noise
            int_array = torch.sqrt(int_array)

            # Step 6: L2-normalize
            # Total ion current (TIC) normalization for cross-sample comparability
            # Creates unit-norm vector preserving relative intensity patterns
            int_array = int_array / torch.linalg.norm(int_array)

            # Step 7: Optionally normalize m/z
            if self.normalize_mz:
                mz_array = mz_array / self.max_mz

            return torch.stack([mz_array, int_array], dim=1).float()

        except ValueError as e:
            logger.debug(f"Spectrum processing failed: {e}")
            # Return dummy spectrum
            dummy_mz = 0.0 if self.normalize_mz else 0.0
            dummy_intensity = 1.0 / self.max_mz if self.normalize_mz else 1.0
            return torch.tensor([[dummy_mz, dummy_intensity]], dtype=torch.float32)

    @staticmethod
    def _pad_and_mask(x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length tensors and create attention masks.
        
        Used for peptide sequences which have dynamic lengths.
        
        Args:
            x: List of tensors with different lengths.
            
        Returns:
            Tuple of (padded_tensor, mask):
                - padded_tensor: [batch_size, max_length, ...]
                - mask: [batch_size, max_length], True for padding positions
        """
        lengths = torch.tensor([y.shape[0] for y in x], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
        mask = torch.arange(padded.shape[1], dtype=torch.long)[None, :] >= lengths[:, None]
        return padded, mask

    def apply_peak_ordering(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply peak reordering to test positional encoding robustness.

        Uses fully vectorized batched tensor operations (no Python loops over B).

        Strategies:
            - sorted: Sort by m/z (default, preserves spectral patterns)
            - cyclic_shift: Random circular rotation per spectrum
            - complete_shuffle: Full randomization per spectrum

        Args:
            spectra: Padded spectra [batch_size, n_peaks, 2].
            spectra_mask: Padding mask [batch_size, n_peaks], True for padding.

        Returns:
            Reordered spectra [batch_size, n_peaks, 2].
        """
        B, L, C = spectra.shape

        # Extract m/z channel for sorting
        mz = spectra[:, :, 0]  # (B, L)

        # Set padding positions to inf so they sort to the end
        mz_for_sort = mz.clone()
        mz_for_sort[spectra_mask] = float("inf")

        # Sort by m/z (ascending) — padding goes to the end
        sort_indices = torch.argsort(mz_for_sort, dim=1)  # (B, L)

        # Gather sorted spectra using batched index
        # Expand indices for gathering across the feature dimension
        gather_idx = sort_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, L, C)
        reordered = torch.gather(spectra, 1, gather_idx)  # (B, L, C)

        if self.peak_ordering == "sorted":
            return reordered

        elif self.peak_ordering == "cyclic_shift":
            # Compute number of valid peaks per spectrum
            n_valid = (~spectra_mask).sum(dim=1)  # (B,)

            # Generate random shifts per spectrum (vectorized)
            # For spectra with <=1 valid peak, shift=0 (no effect)
            max_valid = n_valid.clamp(min=1)
            shifts = (torch.rand(B, device=spectra.device) * max_valid.float()).long()  # (B,)

            # Build shifted index for each spectrum
            positions = torch.arange(L, device=spectra.device).unsqueeze(0).expand(B, -1)  # (B, L)
            # Apply cyclic shift only within valid region, leave padding in place
            shifted_positions = positions.clone()
            for_shift = (positions - shifts.unsqueeze(1)) % max_valid.unsqueeze(1)
            # Only apply shift within the valid region
            valid_region = positions < n_valid.unsqueeze(1)
            shifted_positions[valid_region] = for_shift[valid_region]

            shift_idx = shifted_positions.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(reordered, 1, shift_idx)

        elif self.peak_ordering == "complete_shuffle":
            # Noise-based batched permutation within valid region
            n_valid = (~spectra_mask).sum(dim=1)  # (B,)

            # Generate random noise for valid positions, large values for padding
            noise = torch.rand(B, L, device=spectra.device)
            noise[spectra_mask] = float("inf")  # Padding stays at the end

            # Argsort noise to get per-spectrum permutations
            perm_indices = torch.argsort(noise, dim=1)  # (B, L)
            perm_idx = perm_indices.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(reordered, 1, perm_idx)

        return reordered

    def apply_masking(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
        precursor_charges: torch.Tensor | None = None,
        peptides: list[str] | None = None,
        frag_types: list[str | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking strategy for self-supervised learning.

        Masks ~30% of peaks for reconstruction task. Supports:
            - thompson: Intensity-weighted probabilistic masking
            - thompson_span: Span masking + isotope co-masking (default)
            - uniform: Random baseline
            - signal_aware_span: Signal vs noise aware
            - signal_aware_fragment: Fragment ion family masking (requires sequences)

        Note: The model uses learned mask embeddings (mask_token) to handle masked
        positions. The encoder replaces embeddings at masked positions with the
        learned mask_token, so input values at masked positions are not used.
        This method only computes which positions to mask (peak_mask) and preserves
        targets for loss computation.

        Args:
            spectra: Spectra tensor [batch_size, n_peaks, 2].
            spectra_mask: Padding mask [batch_size, n_peaks].
            precursor_charges: Charge states [batch_size] for isotope detection.
            peptides: Peptide sequences [batch_size] for signal-aware masking (optional).

        Returns:
            Tuple of (peak_mask, targets):
                - peak_mask: [batch_size, n_peaks], True for masked positions
                - targets: Ground truth spectra for reconstruction loss
        """
        if self.masking_strategy == "none":
            return torch.zeros_like(spectra_mask, dtype=torch.bool), spectra.clone()
        
        # Extract m/z and intensity channels
        mz = spectra[:, :, 0]
        intensity = spectra[:, :, 1]
        
        # Get masking function
        mask_fn = get_mask_function(self.masking_strategy)
        
        # Call appropriate masking strategy
        if self.masking_strategy == "thompson":
            peak_mask = mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=self.mask_portion,
                alpha=self.thompson_alpha,
                beta=self.thompson_beta,
                kappa=self.thompson_kappa,
                gamma=self.thompson_gamma,
            )
            
        elif self.masking_strategy == "thompson_span":
            peak_mask = mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=self.mask_portion,
                span_min=self.span_min,
                span_max=self.span_max,
                alpha=self.thompson_alpha,
                beta=self.thompson_beta,
                kappa=self.thompson_kappa,
                gamma=self.thompson_gamma,
                bidirectional=self.span_bidirectional,
                mz=mz,
                charges=precursor_charges if self.include_isotopes else None,
                include_isotopes=self.include_isotopes,
                isotope_ppm=self.isotope_ppm,
                isotope_da_floor=self.isotope_da_floor,
                isotope_max_charge=self.isotope_max_charge,
                isotope_max_order=self.isotope_max_order,
                max_total_mask_ratio=self.max_total_mask_ratio,
                max_mz=self.max_mz,
                normalize_mz=self.normalize_mz,
            )

        elif self.masking_strategy == "uniform":
            peak_mask = mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mask_portion=self.mask_portion,
            )
            
        elif self.masking_strategy == "signal_aware_span":
            peak_mask = mask_fn(
                intensity=intensity,
                spectra_pad=spectra_mask,
                signal_mask=None,  # Can be enhanced with signal detection
                mask_portion=self.mask_portion,
                span_min=self.span_min,
                span_max=self.span_max,
                alpha=self.thompson_alpha,
                beta=self.thompson_beta,
                kappa=self.thompson_kappa,
                gamma=self.thompson_gamma,
                bidirectional=self.span_bidirectional,
            )

        elif self.masking_strategy == "signal_aware_fragment":
            peak_mask = mask_fn(
                intensity=intensity,
                spectra_mask=spectra_mask,
                mz=mz,
                charges=precursor_charges,
                peptides=peptides,
                # Signal-aware config
                mask_portion=self.mask_portion,
                min_backbone_coverage=self.signal_min_backbone_coverage,
                min_fragment_groups=self.signal_min_fragment_groups,
                annotation_ppm=self.signal_ppm,
                annotation_cid_da_tol=self.signal_cid_da_tol,
                annotation_ion_types=self.signal_ion_types,
                frag_types=frag_types,
                max_mz=self.max_mz,
                normalize_mz=self.normalize_mz,
                # Fallback config (thompson_span)
                span_min=self.span_min,
                span_max=self.span_max,
                alpha=self.thompson_alpha,
                beta=self.thompson_beta,
                kappa=self.thompson_kappa,
                gamma=self.thompson_gamma,
                bidirectional=self.span_bidirectional,
                include_isotopes=self.include_isotopes,
                isotope_ppm=self.isotope_ppm,
                isotope_da_floor=self.isotope_da_floor,
                isotope_max_charge=self.isotope_max_charge,
                isotope_max_order=self.isotope_max_order,
                max_total_mask_ratio=self.max_total_mask_ratio,
                # Performance
                num_workers=self.signal_num_workers,
            )

        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        # Return peak_mask and targets for loss computation
        # Note: The model uses learned mask_token embeddings, so we don't need
        # to create masked input values - the encoder handles masking internally
        targets = spectra.clone()
        return peak_mask, targets

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process single sample from dataset.
        
        Always processes spectrum data. Optionally processes sequence if
        annotated=True (for evaluation/analysis only).
        
        Args:
            row: Raw data with keys:
                - mz_array: Peak m/z values
                - intensity_array: Peak intensities
                - precursor_mz: Precursor m/z
                - precursor_charge: Precursor charge
                - sequence: Peptide sequence (optional)
                
        Returns:
            Processed data with keys:
                - spectra: [n_peaks, 2] tensor
                - precursor_mz: float
                - precursor_charge: int
                - peptide: string or tensor (if annotated=True)
                - metadata: any additional columns
        """
        processed = {}

        # Process spectrum (always)
        spectra = self._process_spectrum(
            torch.tensor(row[MSColumns.MZ_ARRAY.value]),
            torch.tensor(row[MSColumns.INTENSITY_ARRAY.value]),
            row[MSColumns.PRECURSOR_MZ.value],
            row[MSColumns.PRECURSOR_CHARGE.value],
        )
        processed["spectra"] = spectra
        processed["precursor_mz"] = row[MSColumns.PRECURSOR_MZ.value]
        processed["precursor_charge"] = row[MSColumns.PRECURSOR_CHARGE.value]

        # Process sequence (optional, for evaluation)
        if self.annotated and ANNOTATED_COLUMN in row:
            peptide = row[ANNOTATED_COLUMN]
            if self.return_str:
                processed["peptide"] = peptide
            else:
                # Tokenize sequence
                if self.residue_set is None:
                    raise RuntimeError("residue_set is None but return_str=False")
                tokens = self.residue_set.tokenize(peptide)
                processed["peptide"] = self.residue_set.encode(
                    tokens, add_eos=False, return_tensor="pt"
                )

        # Add metadata from original row
        # Only copy columns that are actually used (performance optimization)
        if self.metadata_columns:
            # For metadata token, we only need: frag_type, collision_energy, + spectrum key (usi/filepath)
            # Other fields (precursor_mz, precursor_charge, precursor_mass) are already in batch
            for col in self.metadata_columns:
                if col in row:
                    processed[col] = row[col]
                elif col not in self._warned_missing_columns:
                    self._warned_missing_columns.add(col)
                    logger.warning(
                        f"Metadata column '{col}' listed in config but absent from data. "
                        f"Will be silently skipped for all rows."
                    )
        return processed

    def _get_expected_columns(self) -> list[str]:
        """Get expected output columns after processing.
        
        Returns columns that should be included in dataset format specification.
        Only includes columns that are present in the processed output from
        process_row(). Dynamic columns added in collate_fn are not included.
        """
        columns = ["spectra", "precursor_mz", "precursor_charge"]
        if self.annotated:
            columns.append("peptide")
        return columns

    def _collate_batch(
        self, 
        batch: list[dict[str, Any]], 
        apply_masking: bool = True
    ) -> dict[str, torch.Tensor | Any]:
        """Collate batch with fixed-length padding and optional masking.
        
        Key Design: Uses FIXED-LENGTH padding to n_peaks (not dynamic).
        Optimal when most spectra have ~200 peaks, providing:
            - Consistent tensor shapes across batches
            - Better GPU optimization (torch.compile friendly)
            - Lower memory usage than dynamic padding
        
        Pipeline:
            1. Extract data from batch
            2. Pad all spectra to fixed length n_peaks
            3. Create padding masks
            4. Combine precursor information
            5. Apply peak ordering (optional reordering)
            6. Apply masking strategy (for self-supervised learning)
            7. Collate sequences (for evaluation if annotated=True)
        
        Args:
            batch: List of processed samples from process_row.
            apply_masking: Whether to apply masking (default: True).
            
        Returns:
            Dictionary with keys:
                - spectra: [B, n_peaks, 2]
                - spectra_mask: [B, n_peaks], True for padding
                - precursors: [B, 3], [mass, charge, mz]
                
                If apply_masking=True:
                - masked_spectra: [B, n_peaks, 2], masked input
                - peak_mask: [B, n_peaks], True for masked
                - targets: [B, n_peaks, 2], ground truth
                
                If annotated=True:
                - peptides: list of strings or tensor
                - peptides_mask: mask if tokenized
                
                Plus any metadata columns as lists.
        """
        # Extract spectra and precursor info
        data_batch = [
            (row["spectra"], row["precursor_mz"], row["precursor_charge"])
            for row in batch
        ]
        spectra, precursor_mzs, precursor_charges = zip(*data_batch, strict=True)

        # Fixed-length padding to n_peaks (consistent shapes)
        B = len(spectra)
        L_static = self.n_peaks
        
        # Track true lengths before padding
        true_lengths = torch.tensor([x.shape[0] for x in spectra], dtype=torch.long)
        
        # Allocate fixed-size tensor (padding value: 0.0)
        spectra_padded = torch.zeros((B, L_static, 2), dtype=torch.float32)
        for i, spectrum in enumerate(spectra):
            length = min(spectrum.shape[0], L_static)  # Handle truncation
            spectra_padded[i, :length] = spectrum[:length]
        
        # Create padding mask (True for padded positions)
        spectra_mask = torch.arange(L_static, dtype=torch.long)[None, :] >= true_lengths[:, None]

        # Combine precursor info: [mass, charge, mz]
        precursor_mzs = torch.tensor(precursor_mzs)
        precursor_charges = torch.tensor(precursor_charges)
        precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

        # Ensure contiguous memory layout (better performance)
        precursors = precursors.contiguous()
        spectra_padded = spectra_padded.contiguous()
        spectra_mask = spectra_mask.contiguous()

        # Apply peak ordering strategy
        spectra_ordered = self.apply_peak_ordering(spectra_padded, spectra_mask)
        spectra_ordered = spectra_ordered.contiguous()

        # Prepare return batch
        return_batch: dict[str, Any] = {
            "spectra": spectra_ordered,
            "precursors": precursors,
            "spectra_mask": spectra_mask,
            # Add individual precursor fields for metadata access
            "precursor_mz": precursor_mzs,
            "precursor_charge": precursor_charges,
            "precursor_mass": precursor_masses,  # Add precursor mass for metadata analysis
        }
        
        # Enrich batch rows with search data BEFORE building metadata dict.
        # This ensures search-derived fields (instrument, acquisition, detector,
        # enzyme, quant) are available to build_metadata_dict for meta token encoding.
        if self.search_data_manager and self.search_data_manager.is_loaded:
            # Only request columns needed for metadata token (performance optimization)
            # These columns map to the metadata token features in metadata_builder.py
            required_columns = [
                "project",         # -> search_project
                "instrument",      # -> search_instrument
                "acquisition",     # -> search_acquisition
                "detector",        # -> search_detector
                "fragmentation",   # -> search_fragmentation
                "enzyme",          # -> search_enzyme
                "quant",           # -> search_quant
                "organism",        # -> search_organism
                "modifications",   # -> search_modifications
            ]

            # Batch lookup with column filtering
            spectrum_key = self.search_data_manager.spectrum_filepath_key
            filepaths = [row.get(spectrum_key) or "" for row in batch]  # Convert None to empty string
            search_metadata_list = self.search_data_manager.get_metadata_batch(
                filepaths,
                columns=required_columns
            )

            # Enrich each batch row with search data so build_metadata_dict can see them
            for row, search_meta in zip(batch, search_metadata_list):
                for col in required_columns:
                    val = search_meta.get(col)
                    if val is not None:
                        row[f"search_{col}"] = val

            # Also add to return_batch for downstream analysis
            return_batch["search_project"] = [m.get("project") for m in search_metadata_list]
            return_batch["search_instrument"] = [m.get("instrument") for m in search_metadata_list]
            return_batch["search_acquisition"] = [m.get("acquisition") for m in search_metadata_list]
            return_batch["search_detector"] = [m.get("detector") for m in search_metadata_list]
            return_batch["search_enzyme"] = [m.get("enzyme") for m in search_metadata_list]
            return_batch["search_quant"] = [m.get("quant") for m in search_metadata_list]
            return_batch["search_fragmentation"] = [m.get("fragmentation") for m in search_metadata_list]
            return_batch["search_organism"] = [m.get("organism") for m in search_metadata_list]
            return_batch["search_modifications"] = [m.get("modifications") for m in search_metadata_list]

        # Build metadata dictionary for meta token (only if enabled)
        # Must happen AFTER search data enrichment so all metadata fields are available
        if self.build_metadata:
            from instanovo_fm.data.metadata_builder import build_metadata_dict
            meta = build_metadata_dict(batch, device=spectra_ordered.device)
            return_batch["meta"] = meta if meta else None

        # Extract peptides and frag_types for signal-aware masking (if available and needed)
        peptides_for_masking = None
        frag_types_for_masking = None
        if self.masking_strategy == "signal_aware_fragment":
            if self.annotated:
                # Extract peptides from batch if they exist
                peptides_batch = [row.get("peptide", None) for row in batch]
                # Only pass if they're strings (not tokenized tensors)
                if peptides_batch and isinstance(peptides_batch[0], (str, type(None))):
                    peptides_for_masking = peptides_batch
            # Extract fragmentation types for Da-aware CID tolerance
            frag_types_for_masking = [row.get("frag_type", None) for row in batch]

        # Apply masking for self-supervised learning
        # Note: The model uses learned mask_token embeddings, so we pass original
        # spectra to the model. The encoder replaces embeddings at masked positions
        # with the learned mask_token internally.
        if apply_masking:
            peak_mask, targets = self.apply_masking(
                spectra=spectra_ordered,
                spectra_mask=spectra_mask,
                precursor_charges=precursor_charges,
                peptides=peptides_for_masking,
                frag_types=frag_types_for_masking,
            )
            return_batch["peak_mask"] = peak_mask.contiguous()
            return_batch["targets"] = targets.contiguous()

        # Add sequences if annotated (for evaluation)
        if self.annotated and "peptide" in batch[0]:
            peptides_batch = [row["peptide"] for row in batch]

            if isinstance(peptides_batch[0], str):
                # Keep as strings (don't add mask for strings)
                return_batch["peptides"] = peptides_batch
            else:
                # Pad tokenized sequences
                peptides, peptides_mask = FoundationalDataProcessor._pad_and_mask(peptides_batch)
                return_batch["peptides"] = peptides.contiguous()
                return_batch["peptides_mask"] = peptides_mask.contiguous()

        # Collate metadata columns from individual processed rows.
        # Numeric columns are converted to tensors for Accelerate broadcast
        # compatibility (NCCL cannot broadcast plain Python floats/ints).
        # String columns and columns with None values remain as lists.
        if self.metadata_columns:
            for col in self.metadata_columns:
                col_data = [row.get(col, None) for row in batch]
                # Only convert to tensor if ALL values are numeric (no Nones)
                if col_data and all(isinstance(v, (int, float)) for v in col_data):
                    return_batch[col] = torch.tensor(col_data)
                else:
                    return_batch[col] = col_data

        # Strip non-tensor items before returning. accelerate's
        # dispatch_batches=True broadcasts tensor metadata (shape, dtype) to
        # non-root ranks and reconstructs empty tensors there — it cannot
        # handle plain Python lists or strings (causes TypeError in
        # initialize_tensors). Masking has already consumed peptides and
        # frag_types above; search/metadata strings are not used by the
        # training or validation loops.
        # Set self._keep_non_tensor_metadata = True on the processor instance
        # before creating a dataloader when string/list metadata must flow
        # through to embedding_io (e.g., standalone evaluation tasks that need
        # frag_type, sequence, search_instrument, collision_energy, etc.).
        if not getattr(self, '_keep_non_tensor_metadata', False):
            return_batch = {k: v for k, v in return_batch.items() if isinstance(v, torch.Tensor)}

        return return_batch
