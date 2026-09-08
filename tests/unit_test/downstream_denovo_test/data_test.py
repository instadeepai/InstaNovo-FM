from __future__ import annotations

import numpy as np
import torch
from instanovo.constants import ANNOTATED_COLUMN, MSColumns
from instanovo.utils.residues import ResidueSet

from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor


def _make_row(
    mz: list[float],
    intensity: list[float],
    precursor_mz: float = 400.0,
    precursor_charge: int = 2,
    sequence: str | None = None,
) -> dict:
    row = {
        MSColumns.MZ_ARRAY.value: np.asarray(mz, dtype=np.float32),
        MSColumns.INTENSITY_ARRAY.value: np.asarray(intensity, dtype=np.float32),
        MSColumns.PRECURSOR_MZ.value: precursor_mz,
        MSColumns.PRECURSOR_CHARGE.value: precursor_charge,
    }
    if sequence is not None:
        row[ANNOTATED_COLUMN] = sequence
    return row


def test_processor_defaults(denovo_residue_set: ResidueSet) -> None:
    """Verify default attribute values for DownstreamDeNovoDataProcessor."""
    processor = DownstreamDeNovoDataProcessor(residue_set=denovo_residue_set)
    assert processor.n_peaks == 200
    assert processor.min_mz == 50.0
    assert processor.max_mz == 2500.0
    assert processor.annotated is True


def test_process_row_unannotated(denovo_residue_set: ResidueSet) -> None:
    """Unannotated processing returns spectra and precursor info only."""
    processor = DownstreamDeNovoDataProcessor(
        residue_set=denovo_residue_set,
        n_peaks=10,
        min_mz=50.0,
        max_mz=1000.0,
        use_spectrum_utils=False,
        annotated=False,
    )
    row = _make_row(mz=[100.0, 200.0, 300.0], intensity=[1.0, 0.5, 0.25])
    out = processor.process_row(row)
    assert set(out.keys()) == {"spectra", "precursor_mz", "precursor_charge"}
    assert out["spectra"].shape[1] == 2
    assert out["spectra"].shape[0] <= 10


def test_process_row_annotated_tokenizes(denovo_residue_set: ResidueSet) -> None:
    """Annotated processing tokenizes and reverses the peptide sequence."""
    processor = DownstreamDeNovoDataProcessor(
        residue_set=denovo_residue_set,
        n_peaks=10,
        min_mz=50.0,
        max_mz=1000.0,
        use_spectrum_utils=False,
        annotated=True,
        return_str=False,
        reverse_peptide=True,
        add_eos=True,
    )
    row = _make_row(mz=[100.0, 200.0], intensity=[1.0, 0.5], sequence="ABC")
    out = processor.process_row(row)
    assert "peptide" in out
    peptide = out["peptide"]
    assert isinstance(peptide, torch.Tensor)
    # Reversed + EOS-terminated → last token is EOS.
    assert int(peptide[-1].item()) == denovo_residue_set.EOS_INDEX
    # Reversed: C, B, A, EOS
    expected = [
        denovo_residue_set.residue_to_index["C"],
        denovo_residue_set.residue_to_index["B"],
        denovo_residue_set.residue_to_index["A"],
        denovo_residue_set.EOS_INDEX,
    ]
    assert peptide.tolist() == expected


def test_process_spectrum_filters_mz_range(tiny_denovo_processor: DownstreamDeNovoDataProcessor) -> None:
    """Peaks outside [min_mz, max_mz] are removed."""
    mz = torch.tensor([10.0, 100.0, 500.0, 5000.0])
    intensity = torch.tensor([1.0, 1.0, 1.0, 1.0])
    spec = tiny_denovo_processor._process_spectrum(mz, intensity, precursor_mz=400.0, precursor_charge=2)
    # Only the two in-range peaks remain; m/z values are raw (normalize_mz=False).
    assert spec.shape == torch.Size([2, 2])
    assert torch.allclose(spec[:, 0], torch.tensor([100.0, 500.0]))


def test_process_spectrum_topk_and_l2_norm(tiny_denovo_processor: DownstreamDeNovoDataProcessor) -> None:
    """Top-k filtering keeps n_peaks and L2-normalizes intensities."""
    mz = torch.linspace(100.0, 900.0, 12)
    intensity = torch.arange(1, 13, dtype=torch.float32)  # 1..12
    spec = tiny_denovo_processor._process_spectrum(mz, intensity, precursor_mz=50.0, precursor_charge=2)
    assert spec.shape == torch.Size([10, 2])
    # Intensities are √-scaled then L2-normalised, so their norm is 1.
    assert torch.isclose(torch.linalg.norm(spec[:, 1]), torch.tensor(1.0), atol=1e-5)


def test_collate_batch_shapes(tiny_denovo_processor: DownstreamDeNovoDataProcessor) -> None:
    """Collated batch has correct tensor shapes and masks."""
    rows = [
        tiny_denovo_processor.process_row(
            _make_row(mz=[100.0, 200.0, 300.0], intensity=[1.0, 0.5, 0.25], sequence="AB"),
        ),
        tiny_denovo_processor.process_row(
            _make_row(mz=[150.0, 250.0], intensity=[0.8, 0.6], sequence="CD"),
        ),
    ]
    batch = tiny_denovo_processor._collate_batch(rows)
    assert batch["spectra"].shape == torch.Size([2, 10, 2])
    assert batch["spectra_mask"].shape == torch.Size([2, 10])
    assert batch["precursors"].shape == torch.Size([2, 3])
    # Annotated → peptides tensor present.
    assert batch["peptides"].ndim == 2
    assert batch["peptides_mask"].shape == batch["peptides"].shape


def test_expected_columns_respects_annotated(denovo_residue_set: ResidueSet) -> None:
    """Expected columns include peptide only when annotated=True."""
    unannotated = DownstreamDeNovoDataProcessor(residue_set=denovo_residue_set, annotated=False)
    annotated = DownstreamDeNovoDataProcessor(residue_set=denovo_residue_set, annotated=True)
    assert unannotated._get_expected_columns() == ["spectra", "precursor_mz", "precursor_charge"]
    assert annotated._get_expected_columns() == ["spectra", "precursor_mz", "precursor_charge", "peptide"]
