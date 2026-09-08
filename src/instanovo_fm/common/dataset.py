from __future__ import annotations

import logging
import re
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from datasets import Dataset
from torch import nn

from instanovo_fm.common._constants import IN_CODE_TO_MASS, NTERM_ONLY_UNIMOD, UNIMOD_MASS_FALLBACK

logger = logging.getLogger(__name__)


class DataProcessor(metaclass=ABCMeta):
    """Data processor abstract class.

    This class is used to process the data before it is used in the model.
    It is designed to be used with the `Dataset` class from the HuggingFace `datasets` library.

    It includes two main methods:
    - `process_row`: Processes a row of data.
    - `collate_fn`: Collates a batch of data. To be passed to the `DataLoader` class.

    Additionally, it includes a way to pass metadata columns that will be kept after processing a dataset.
    These metadata columns will also bypass the `collate_fn`.
    """

    @property
    def metadata_columns(self) -> set[str]:
        """Get the metadata columns.

        These columns are kept after processing a dataset.

        Returns:
            list[str]: The metadata columns.
        """
        return self._metadata_columns

    def __init__(self, metadata_columns: list[str] | set[str] | None = None):
        """Initialize the data processor.

        Args:
            metadata_columns: The metadata columns to add to the expected columns.
        """
        self._metadata_columns: set[str] = set(metadata_columns or [])

    @abstractmethod
    def _get_expected_columns(self) -> list[str]:
        """Get the expected columns.

        These are the columns that will be returned by the `process_row` method.

        Returns:
            list[str]: The expected columns.
        """
        ...

    @abstractmethod
    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of data.

        Args:
            row (dict[str, Any]): The row of data to process in dict format.

        Returns:
            dict[str, Any]: The processed row with resulting columns.
        """
        ...

    @abstractmethod
    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Logic for collating a batch.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any]: The collated batch.
        """
        ...

    def process_dataset(self, dataset: Dataset, return_format: str | None = "torch") -> Dataset:
        """Process a dataset by mapping the `process_row` method.

        The resulting dataset has the columns expected by the `collate_fn` method.

        Args:
            dataset (Dataset): The dataset to process.
            return_format (str | None): The format to return the dataset in.
                Default is "torch".

        Returns:
            Dataset: The processed dataset.
        """
        import hashlib
        import os

        # Provide a random fingerprint to skip update_fingerprint(), which
        # pickles the function closure via dill and causes MemoryError on
        # large Arrow-backed datasets.
        new_fp = hashlib.md5(os.urandom(16)).hexdigest()
        dataset = dataset.map(self.process_row, load_from_cache_file=False, new_fingerprint=new_fp, writer_batch_size=100)
        dataset.set_format(type=return_format, columns=self.get_expected_columns())
        return dataset

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch.

        Metadata columns are added after collation.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any]: The collated batch with metadata.
        """
        return_batch: dict[str, Any] = self._collate_batch(batch)

        # Add metadata columns as tensors where possible.
        # accelerate's dispatch_batches=True can only broadcast tensors —
        # non-tensor types (str, None, lists) cause TypeError in
        # _gpu_broadcast_one. Two cases:
        #   1. Map-style dataset: batch = [row0, row1, ...]; each row[col] is a
        #      scalar (Python int/float or np.int32). Build a 1-D tensor.
        #   2. Pre-batching iterable (SpectrumDataFrame): batch = [batch_dict];
        #      batch_dict[col] is already a tensor of shape (B,). Use directly.
        # Variable-length per-peak fields (intensity_array, mz_array) are also
        # tensors with numel() > 1 but their length equals total peaks, not B.
        # Guard against these by checking the tensor length matches batch size.
        # Strings and unconvertible types are silently dropped.
        n_spectra = return_batch["spectra"].shape[0]
        for col in self.metadata_columns:
            if col in return_batch:
                continue
            col_data: list[Any] = [row[col] if col in row else None for row in batch]
            if not col_data or any(v is None for v in col_data):
                continue
            try:
                first = col_data[0]
                if isinstance(first, torch.Tensor) and first.numel() > 1:
                    if first.shape[0] != n_spectra:
                        # Variable-length per-peak field — not broadcastable, drop.
                        continue
                    # Pre-batched per-spectrum tensor (e.g. prediction_id).
                    return_batch[col] = first if len(col_data) == 1 else torch.cat(col_data, dim=0)
                else:
                    return_batch[col] = torch.tensor(
                        [v.item() if hasattr(v, "item") else v for v in col_data]  # type: ignore[union-attr]
                    )
            except (TypeError, ValueError, RuntimeError):
                pass  # string, mixed-type, or unconvertible — drop

        return return_batch

    def get_expected_columns(self) -> list[str]:
        """Get the expected columns to be kept in the dataset after processing.

        These columns are expected by the `collate_fn` method and include
        both data and metadata columns.

        Returns:
            list[str]: The expected columns.
        """
        return self._get_expected_columns() + list(self.metadata_columns)

    def add_metadata_columns(self, columns: list[str] | set[str]) -> None:
        """Add expected metadata columns.

        Args:
            columns (list[str] | set[str]): The columns to add.
        """
        self._metadata_columns.update(set(columns))

    @staticmethod
    def _pad_and_mask(
        tensor_list: list[torch.tensor] | tuple[torch.tensor],
    ) -> tuple[torch.tensor, torch.tensor]:
        """Pad and mask a list of tensors.

        Args:
            tensor_list (list[torch.tensor] | tuple[torch.tensor]): The list of tensors to pad and mask.

        Returns:
            tuple[torch.tensor, torch.tensor]: The padded and masked tensors.
        """
        ll = torch.tensor([y.shape[0] for y in tensor_list], dtype=torch.long)
        padded_tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
        attention_mask = torch.arange(padded_tensor.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
        return padded_tensor, attention_mask

    @staticmethod
    def remove_modifications(peptide: str, replace_isoleucine_with_leucine: bool = True) -> str:
        """Remove modifications and optionally replace Isoleucine with Leucine.

        Args:
            peptide (str): The peptide to remove modifications from.
            replace_isoleucine_with_leucine (bool): Whether to replace Isoleucine with Leucine.

        Returns:
            str: The peptide with modifications removed.
        """
        # Strip all bracket contents (e.g. [UNIMOD:35], [IN:3074], [+1234.56])
        # to prevent modification text from being picked up as amino acids.
        peptide = re.sub(r"\[[^\]]*\]", "", peptide)
        # use regex to extract only A-Z
        peptide = re.findall(r"[A-Z]", peptide)
        # replace I with L
        if replace_isoleucine_with_leucine:
            peptide = ["L" if aa == "I" else aa for aa in peptide]
        return "".join(peptide)

    @staticmethod
    def convert_in_codes_to_mass_shifts(seq: str) -> str:
        """Convert [IN:XXXX] glycan codes to [+MASS] format for PyOpenMS.

        Uses the IN_CODE_TO_MASS lookup table from constants.py.
        Unknown IN codes are logged as warnings and left unchanged.

        Args:
            seq: Peptide sequence potentially containing [IN:XXXX] modifications.

        Returns:
            Sequence with [IN:XXXX] replaced by [+MASS] mass shifts.
        """

        def _replace_in_code(match: re.Match) -> str:
            code = match.group(1)  # e.g. "IN:3032"
            if code in IN_CODE_TO_MASS:
                return f"[+{IN_CODE_TO_MASS[code]:.4f}]"
            logger.warning("Unknown IN code [%s] in sequence, leaving unchanged.", code)
            return match.group(0)  # type: ignore[no-any-return]

        return re.sub(r"\[(IN:\d+)\]", _replace_in_code, seq)

    @staticmethod
    def convert_unimod_to_pyopenms(seq: str) -> str:
        """Convert [UNIMOD:x] format to PyOpenMS-compatible (UniMod:x) format.

        PyOpenMS requires specific formatting for modified peptides:
        - N-terminal: .(UniMod:x)PEPTIDE.
        - C-terminal: .PEPTIDE(UniMod:x).
        - Internal: .X(UniMod:x)EPTIDE.

        Args:
            seq: Peptide sequence with UNIMOD modifications (e.g., "M[UNIMOD:35]PEPTIDE")

        Returns:
            Sequence in PyOpenMS-compatible format (e.g., ".M(UniMod:35)EPTIDE.")

        Example:
            >>> convert_unimod_to_pyopenms("M[UNIMOD:35]PEPTIDE")
            ".M(UniMod:35)EPTIDE."
            >>> convert_unimod_to_pyopenms("[UNIMOD:1]MPEPTIDE[UNIMOD:4]")
            ".(UniMod:1)MPEPTIDE(UniMod:4)."
        """
        if not seq or not isinstance(seq, str):
            return seq

        # Add .dot delimiters for N/C-termini if missing
        if not seq.startswith("."):
            seq = "." + seq
        if not seq.endswith("."):
            seq += "."

        # Helper: format a UniMod ID as PyOpenMS annotation.  IDs that
        # PyOpenMS doesn't recognise (UNIMOD_MASS_FALLBACK) are emitted as
        # mass-shift notation [+MASS] which PyOpenMS handles dynamically.
        def _fmt(uid: int) -> str:
            if uid in UNIMOD_MASS_FALLBACK:
                return f"[+{UNIMOD_MASS_FALLBACK[uid]:.6f}]"
            return f"(UniMod:{uid})"

        # Handle [UNIMOD:x] at the N-terminus (no preceding amino acid).
        # N-terminal-only mods (e.g. 1=Acetyl, 27/28=pyro-Glu) stay as
        # N-terminal syntax: .(UniMod:x)AA...
        # All other mods (e.g. 21=Phospho) are moved onto the first amino
        # acid: .AA(UniMod:x)... because PyOpenMS rejects them as N-terminal.
        def _nterm_rewrite(match: re.Match) -> str:
            uid = int(match.group(1))
            after = match.group(2)  # text after the mod (first AA + rest)
            mod_str = _fmt(uid)
            if uid in NTERM_ONLY_UNIMOD:
                return f".{mod_str}{after}"
            # Move mod onto the first amino acid
            if after and after[0].isalpha():
                return f".{after[0]}{mod_str}{after[1:]}"
            # No amino acid follows — keep as N-terminal (best effort)
            return f".{mod_str}{after}"

        seq = re.sub(r"^\.\[UNIMOD:(\d+)\](.*)", _nterm_rewrite, seq, flags=re.IGNORECASE)

        # N-terminal-only mods on the first amino acid: .X[UNIMOD:27] → .(UniMod:27)X
        # This handles the case where the raw annotation has the mod after the
        # first AA (e.g. E[UNIMOD:27]PEPTIDE) but PyOpenMS needs N-terminal syntax.
        def _first_aa_nterm_rewrite(match: re.Match) -> str:
            aa = match.group(1)
            uid = int(match.group(2))
            mod_str = _fmt(uid)
            if uid in NTERM_ONLY_UNIMOD:
                return f".{mod_str}{aa}"
            return f".{aa}{mod_str}"

        seq = re.sub(r"^\.([A-Z])\[UNIMOD:(\d+)\]", _first_aa_nterm_rewrite, seq, flags=re.IGNORECASE)

        # C-terminal mod: [UNIMOD:x] before trailing dot
        def _cterm_rewrite(match: re.Match) -> str:
            uid = int(match.group(1))
            return f"{_fmt(uid)}."

        seq = re.sub(r"\[UNIMOD:(\d+)\]\.$", _cterm_rewrite, seq, flags=re.IGNORECASE)

        # Internal mod: X[UNIMOD:x] → X(UniMod:x) or X[+MASS]
        def _internal_rewrite(match: re.Match) -> str:
            aa = match.group(1)
            uid = int(match.group(2))
            return f"{aa}{_fmt(uid)}"

        seq = re.sub(r"([A-Z])\[UNIMOD:(\d+)\]", _internal_rewrite, seq, flags=re.IGNORECASE)

        return seq

    @staticmethod
    def _add_implicit_cam(sequence: str) -> str:
        """Add Carbamidomethylation to bare Cysteine residues.

        In most proteomics workflows, iodoacetamide alkylation is applied
        as a **fixed** modification to all Cysteine residues during sample
        preparation.  Fixed modifications are typically not annotated in
        peptide sequence strings, but PyOpenMS needs them explicitly to
        compute correct masses.

        This method adds ``(UniMod:4)`` (+57.02 Da) to every ``C`` that
        is not already followed by a modification annotation (``(`` or
        ``[``).

        Args:
            sequence: PyOpenMS-formatted peptide sequence.

        Returns:
            Sequence with implicit CAM annotations added.
        """
        return re.sub(r"C(?![(\[])", "C(UniMod:4)", sequence)

    @staticmethod
    def clean_peptide_for_pyopenms(
        sequence: str,
        keep_modifications: bool = True,
        add_carbamidomethyl: bool = True,
    ) -> str | None:
        """Clean peptide sequence for PyOpenMS theoretical spectrum generation.

        Two modes:
        1. keep_modifications=True: Convert to PyOpenMS-compatible format (default)
        2. keep_modifications=False: Strip all modifications

        When *keep_modifications* is True and *add_carbamidomethyl* is True
        (the default), Carbamidomethylation (UniMod:4, +57.02 Da) is added
        to any bare Cysteine residue that does not already carry an explicit
        modification annotation.  This accounts for the near-universal
        proteomics convention of treating iodoacetamide alkylation as a
        fixed (unannotated) modification.

        Args:
            sequence: Raw peptide sequence (may contain modifications)
            keep_modifications: If True, convert modifications to PyOpenMS format.
                               If False, remove all modifications.
            add_carbamidomethyl: If True (default), add implicit CAM to bare
                Cys residues when keep_modifications is True.  Set to False
                for experiments that did not use iodoacetamide alkylation.

        Returns:
            Cleaned sequence suitable for PyOpenMS, or None if cleaning failed

        Examples:
            >>> clean_peptide_for_pyopenms("M[UNIMOD:35]PEPTCIDE", keep_modifications=True)
            ".M(UniMod:35)EPTC(UniMod:4)IDE."
            >>> clean_peptide_for_pyopenms("PEPTCIDE", keep_modifications=True)
            "PEPTC(UniMod:4)IDE"
            >>> clean_peptide_for_pyopenms("PEPTCIDE", keep_modifications=True, add_carbamidomethyl=False)
            "PEPTCIDE"
            >>> clean_peptide_for_pyopenms("M[UNIMOD:35]PEPTIDE", keep_modifications=False)
            "MPEPTIDE"
        """
        if not isinstance(sequence, str) or len(sequence.strip()) == 0:
            return None

        sequence = sequence.upper().strip()

        # Strip ProForma terminal modification dashes (e.g. [UNIMOD:1]-PEPTIDE
        # or PEPTIDE-[UNIMOD:2]).  ProForma uses '-' to separate terminal mods
        # from the peptide body, but PyOpenMS doesn't recognise this separator.
        sequence = re.sub(r"\]-", "]", sequence)
        sequence = re.sub(r"-\[", "[", sequence)

        # Mode 1: Keep modifications (convert to PyOpenMS format)
        if keep_modifications:
            # Convert [IN:XXXX] glycan codes to [+MASS] mass shifts first,
            # so subsequent format checks see standard notation.
            if re.search(r"\[IN:\d+\]", sequence):
                sequence = DataProcessor.convert_in_codes_to_mass_shifts(sequence)

            # Check for UNIMOD format (may coexist with mass shifts from IN conversion)
            if re.search(r"\[UNIMOD:\d+\]", sequence, flags=re.IGNORECASE):
                sequence = DataProcessor.convert_unimod_to_pyopenms(sequence)

            result = None

            # Check if already in PyOpenMS format (UniMod:X)
            if re.search(r"\(UniMod:\d+\)", sequence, flags=re.IGNORECASE):
                # Already in PyOpenMS format - ensure terminal dots are present
                if not sequence.startswith("."):
                    sequence = "." + sequence
                if not sequence.endswith("."):
                    sequence += "."
                result = sequence

            # Check for PyOpenMS-compatible mass modifications [+/-number]
            elif re.search(r"\[[+-]?\d+\.?\d*\]", sequence):
                # Add terminal dots if missing
                if not sequence.startswith("."):
                    sequence = "." + sequence
                if not sequence.endswith("."):
                    sequence += "."
                result = sequence

            # Check if pure amino acids (no modifications to convert)
            elif re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", sequence):
                result = sequence

            else:
                # Has modifications but in unknown format - try to clean
                logger.warning(
                    "Stripping modifications from '%.60s' -- unrecognized modification format",
                    sequence,
                )
                result = DataProcessor.remove_modifications(sequence, replace_isoleucine_with_leucine=False)

            # Add implicit Carbamidomethylation to bare Cys residues
            if result is not None and add_carbamidomethyl and "C" in result:
                result = DataProcessor._add_implicit_cam(result)

            return result

        # Mode 2: Remove all modifications
        else:
            return DataProcessor.remove_modifications(sequence, replace_isoleucine_with_leucine=False)
