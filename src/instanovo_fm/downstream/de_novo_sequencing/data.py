"""Data processor for downstream de novo peptide sequencing."""

from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np
import spectrum_utils.spectrum as sus
import torch
from jaxtyping import Float
from torch import Tensor

from instanovo.__init__ import console
from instanovo_fm.common import DataProcessor
from instanovo.constants import ANNOTATED_COLUMN, PROTON_MASS_AMU, MSColumns
from instanovo_fm.data.search_data_manager import SearchDataManager
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.residues import ResidueSet

logger = ColorLog(console, __name__).logger

PeakOrdering = Literal["sorted", "cyclic_shift", "complete_shuffle"]


class DownstreamDeNovoDataProcessor(DataProcessor):
    """Data processor for supervised de novo peptide sequencing.

    Combines the foundational model's spectrum processing pipeline (with optional
    m/z normalization and meta token support) with peptide tokenization for
    encoder-decoder training.

    Processing Pipeline:
        1. Spectrum preprocessing (filter, normalize, scale)
        2. Fixed-length padding to n_peaks
        3. M/z-sorted peak ordering
        4. Peptide tokenization and padding
        5. Optional metadata enrichment for meta token

    Args:
        residue_set: ResidueSet for peptide tokenization.
        n_peaks: Fixed number of peaks per spectrum (default: 200).
        min_mz: Minimum m/z value to retain (default: 50.0).
        max_mz: Maximum m/z value to retain (default: 2500.0).
        min_intensity: Minimum intensity threshold (default: 0.01).
        remove_precursor_tol: Precursor removal tolerance in Da (default: 2.0).
        use_spectrum_utils: Use spectrum_utils library for preprocessing (default: True).
        normalize_mz: Normalize m/z to [0,1] by dividing by max_mz (default: False).
        peak_ordering: Peak ordering strategy (default: "sorted").
            - "sorted": Keep m/z order (preserves spectral patterns)
            - "cyclic_shift": Random rotation (data augmentation)
            - "complete_shuffle": Full randomization (hardest test)
        annotated: Include peptide sequence processing (default: True).
        return_str: Keep sequences as strings vs tokenized (default: False).
        reverse_peptide: Reverse peptide sequence for auto-regressive decoding (default: True).
        add_eos: Add end-of-sequence token to tokenized peptides (default: True).
        metadata_columns: Additional columns to include in batches.
        search_data_manager: Optional SearchDataManager for additional metadata.
        build_metadata: Build metadata dict for meta token (default: False).
    """

    def __init__(
        self,
        residue_set: ResidueSet,
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 1e-6,
        remove_precursor_tol: float = 2.0,
        use_spectrum_utils: bool = True,
        normalize_mz: bool = False,
        peak_ordering: PeakOrdering = "sorted",
        annotated: bool = True,
        return_str: bool = False,
        reverse_peptide: bool = True,
        add_eos: bool = True,
        metadata_columns: list[str] | None = None,
        search_data_manager: SearchDataManager | None = None,
        build_metadata: bool = False,
    ) -> None:
        """Initialize the downstream de novo data processor."""
        super().__init__(metadata_columns=metadata_columns)

        # Spectrum processing config
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.use_spectrum_utils = use_spectrum_utils
        self.normalize_mz = normalize_mz
        self.peak_ordering = peak_ordering

        # Sequence processing config
        self.residue_set = residue_set
        self.annotated = annotated
        self.return_str = return_str
        self.reverse_peptide = reverse_peptide
        self.add_eos = add_eos

        # Search data and metadata config
        self.search_data_manager = search_data_manager
        self.build_metadata = build_metadata

        search_data_status = "enabled" if self.search_data_manager else "disabled"
        logger.info(f"DownstreamDeNovoDataProcessor initialized: n_peaks={n_peaks}, search_data={search_data_status}")

    def apply_peak_ordering(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply peak reordering to test positional encoding robustness.

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
        batch_size = spectra.shape[0]
        reordered = spectra.clone()

        for b in range(batch_size):
            valid_mask = ~spectra_mask[b]
            n_valid = int(valid_mask.sum().item())

            if n_valid <= 1:
                continue

            if self.peak_ordering == "sorted":
                # Sort by m/z (ascending)
                valid_spectra = spectra[b, :n_valid]
                mz_values = valid_spectra[:, 0]
                sort_indices = torch.argsort(mz_values)
                reordered[b, :n_valid] = valid_spectra[sort_indices]

            elif self.peak_ordering == "cyclic_shift":
                # First sort by m/z, then apply random rotation
                valid_spectra = spectra[b, :n_valid]
                mz_values = valid_spectra[:, 0]
                sort_indices = torch.argsort(mz_values)
                sorted_spectra = valid_spectra[sort_indices]

                # Random rotation
                shift = int(torch.randint(0, n_valid, (1,)).item())
                reordered[b, :n_valid] = torch.roll(sorted_spectra, shifts=shift, dims=0)

            elif self.peak_ordering == "complete_shuffle":
                # Sort by m/z first, then shuffle
                valid_spectra = spectra[b, :n_valid]
                mz_values = valid_spectra[:, 0]
                sort_indices = torch.argsort(mz_values)
                sorted_spectra = valid_spectra[sort_indices]

                # Full randomization
                perm = torch.randperm(n_valid)
                reordered[b, :n_valid] = sorted_spectra[perm]

        return reordered

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

                # Scale and normalize
                spectrum.scale_intensity("root", 1)  # √intensity
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
            int_array = torch.sqrt(int_array)  # TODO update with new data proc

            # Step 6: L2-normalize
            int_array = int_array / torch.linalg.norm(int_array)  # TODO update with new data proc

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

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process single sample from dataset.

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

        # Spectra processing
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
                tokens = self.residue_set.tokenize(peptide)

                if self.reverse_peptide:
                    tokens = tokens[::-1]

                processed["peptide"] = self.residue_set.encode(tokens, add_eos=self.add_eos, return_tensor="pt")

        if self.metadata_columns:
            for col in self.metadata_columns:
                if col in row:
                    processed[col] = row[col]
        return processed

    def _get_expected_columns(self) -> list[str]:
        """Get expected output columns after processing."""
        columns = ["spectra", "precursor_mz", "precursor_charge"]
        if self.annotated:
            columns.append("peptide")
        return columns

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        """Collate batch with fixed-length padding.

        Args:
            batch: List of processed samples from process_row.

        Returns:
            Dictionary with keys:
                - spectra: [B, n_peaks, 2]
                - spectra_mask: [B, n_peaks], True for padding
                - precursors: [B, 3], [mass, charge, mz]
                - peptides: tensor or list of strings (if annotated)
                - peptides_mask: [B, max_seq_len] (if tokenized)
                - meta: metadata dict (if build_metadata enabled)
                Plus any metadata columns as lists.
        """
        data_batch = [(row["spectra"], row["precursor_mz"], row["precursor_charge"]) for row in batch]
        spectra, precursor_mzs, precursor_charges = zip(*data_batch, strict=True)

        batch_size = len(spectra)
        n_peaks = self.n_peaks

        true_lengths = torch.tensor([x.shape[0] for x in spectra], dtype=torch.long)

        spectra_padded = torch.zeros((batch_size, n_peaks, 2), dtype=torch.float32)
        for i, spectrum in enumerate(spectra):
            length = min(spectrum.shape[0], n_peaks)
            spectra_padded[i, :length] = spectrum[:length]

        spectra_mask = torch.arange(n_peaks, dtype=torch.long)[None, :] >= true_lengths[:, None]

        precursor_mzs = torch.tensor(precursor_mzs)
        precursor_charges = torch.tensor(precursor_charges)
        precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

        precursors = precursors.contiguous()
        spectra_padded = spectra_padded.contiguous()
        spectra_mask = spectra_mask.contiguous()

        spectra_ordered = self.apply_peak_ordering(spectra_padded, spectra_mask)
        spectra_ordered = spectra_ordered.contiguous()

        return_batch: dict[str, Any] = {
            "spectra": spectra_ordered,
            "precursors": precursors,
            "spectra_mask": spectra_mask,
            "precursor_mz": precursor_mzs,
            "precursor_charge": precursor_charges,
            "precursor_mass": precursor_masses,
        }

        if self.search_data_manager and self.search_data_manager.is_loaded:
            required_columns = [
                "project",
                "instrument",
                "acquisition",
                "detector",
                "fragmentation",
                "enzyme",
                "quant",
                "organism",
                "modifications",
            ]

            filepaths = [row.get("filepath") or "" for row in batch]
            search_metadata_list = self.search_data_manager.get_metadata_batch(filepaths, columns=required_columns)

            for row, search_meta in zip(batch, search_metadata_list, strict=False):
                for col in required_columns:
                    val = search_meta.get(col)
                    if val is not None:
                        row[f"search_{col}"] = val

            return_batch["search_project"] = [m.get("project") for m in search_metadata_list]
            return_batch["search_instrument"] = [m.get("instrument") for m in search_metadata_list]
            return_batch["search_acquisition"] = [m.get("acquisition") for m in search_metadata_list]
            return_batch["search_detector"] = [m.get("detector") for m in search_metadata_list]
            return_batch["search_enzyme"] = [m.get("enzyme") for m in search_metadata_list]
            return_batch["search_quant"] = [m.get("quant") for m in search_metadata_list]
            return_batch["search_fragmentation"] = [m.get("fragmentation") for m in search_metadata_list]
            return_batch["search_organism"] = [m.get("organism") for m in search_metadata_list]
            return_batch["search_modifications"] = [m.get("modifications") for m in search_metadata_list]

        if self.build_metadata:
            from instanovo_fm.data.metadata_builder import build_metadata_dict

            meta = build_metadata_dict(batch, device=spectra_ordered.device)
            return_batch["meta"] = meta if meta else None

        # Add sequences if annotated (for evaluation)
        if self.annotated:
            peptides_batch = [row["peptide"] for row in batch]

            if isinstance(peptides_batch[0], str):
                return_batch["peptides"] = peptides_batch
                return_batch["peptides_mask"] = None
            else:
                peptides, peptides_mask = DataProcessor._pad_and_mask(peptides_batch)

                return_batch["peptides"] = peptides.contiguous()
                return_batch["peptides_mask"] = peptides_mask.contiguous()

        # Collate metadata columns from individual processed rows
        if self.metadata_columns:
            for col in self.metadata_columns:
                return_batch[col] = [row.get(col, None) for row in batch]

        return_batch = {k: v for k, v in return_batch.items() if isinstance(v, torch.Tensor)}

        return return_batch
