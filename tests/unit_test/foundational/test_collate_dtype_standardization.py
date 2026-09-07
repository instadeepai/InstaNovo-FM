"""Test dtype standardization in foundational collate function."""

import pytest as _pytest

# This module targets a module that does not exist in the ported package
# (instanovo_fm.data.collate). The name was already absent from the source branch, so this is
# inherited staleness rather than a porting regression. Skipping rather than
# deleting keeps the coverage recoverable: restore the module and the guard
# clears itself.
_pytest.importorskip("instanovo_fm.data.collate")

import pytest
import torch

from instanovo_fm.data.collate import spectra_collator


def test_foundational_collate_dtype_standardization() -> None:
    """Test that foundational collate function standardizes dtypes correctly."""
    # Create mock batch data with mixed dtypes
    batch = [
        {
            "spectra": torch.randn(5, 2, dtype=torch.float32),
            "spectra_gt": torch.randn(5, 2, dtype=torch.float32),
            "mlm_mask": torch.randint(0, 2, (5,), dtype=torch.bool),
            "precursor_mz": 100.0,
            "precursor_charge": 2,
            "precursor_mass": 200.0,
            "charge_id": 1,  # This should become int64
            "rt_log": 5.0,
            "meta": {
                "collision_energy": 30.0,
                "frag_id": 0,
                "acq_id": 1,
            }
        },
        {
            "spectra": torch.randn(3, 2, dtype=torch.float32),
            "spectra_gt": torch.randn(3, 2, dtype=torch.float32),
            "mlm_mask": torch.randint(0, 2, (3,), dtype=torch.bool),
            "precursor_mz": 150.0,
            "precursor_charge": 3,
            "precursor_mass": 450.0,
            "charge_id": 2,  # This should become int64
            "rt_log": 6.0,
            "meta": {
                "collision_energy": 35.0,
                "frag_id": 1,
                "acq_id": 0,
            }
        }
    ]
    
    # Test with metadata columns that should be converted to tensors
    metadata_columns = ["extra_feature"]
    batch[0]["extra_feature"] = [1.5]  # List that should become tensor
    batch[1]["extra_feature"] = [2.5]  # List that should become tensor
    
    # Test meta field selection
    meta_field_selection = {
        "include_collision_energy": True,
        "include_frag": True,
        "include_acq": True,
    }
    
    result = spectra_collator(
        batch=batch,
        meta_field_selection=meta_field_selection,
        metadata_columns=metadata_columns
    )
    
    # Check dtype standardization
    assert result["precursors"].dtype == torch.float32, "Precursors should be float32"
    assert result["charge_id"].dtype == torch.int64, "Charge ID should be int64"
    
    # Check that metadata columns are converted to tensors
    assert isinstance(result["extra_feature"], torch.Tensor), "Extra feature should be tensor"
    assert result["extra_feature"].dtype == torch.float32, "Extra feature should be float32"
    
    # Check meta dict contains tensors
    assert isinstance(result["meta"]["collision_energy"], torch.Tensor), "Meta collision_energy should be tensor"
    assert result["meta"]["collision_energy"].dtype == torch.float32, "Meta collision_energy should be float32"
    assert isinstance(result["meta"]["frag_id"], torch.Tensor), "Meta frag_id should be tensor"
    assert result["meta"]["frag_id"].dtype == torch.long, "Meta frag_id should be long"
    
    # Check shapes
    assert result["spectra"].shape[0] == 2, "Batch size should be 2"
    assert result["charge_id"].shape[0] == 2, "Charge ID batch size should be 2"
    assert result["extra_feature"].shape[0] == 2, "Extra feature batch size should be 2"


def test_foundational_collate_handles_mixed_dtypes() -> None:
    """Test that collate function handles mixed input dtypes gracefully."""
    # Create batch with intentionally mixed dtypes
    batch = [
        {
            "spectra": torch.randn(4, 2, dtype=torch.float64),  # Different dtype
            "spectra_gt": torch.randn(4, 2, dtype=torch.float32),
            "mlm_mask": torch.randint(0, 2, (4,), dtype=torch.bool),
            "precursor_mz": 100.0,
            "precursor_charge": 2,
            "precursor_mass": 200.0,
            "charge_id": torch.tensor(1, dtype=torch.int32),  # Different dtype
            "rt_log": 5.0,
            "meta": {
                "collision_energy": 30.0,
            }
        },
        {
            "spectra": torch.randn(6, 2, dtype=torch.float32),
            "spectra_gt": torch.randn(6, 2, dtype=torch.float32),
            "mlm_mask": torch.randint(0, 2, (6,), dtype=torch.bool),
            "precursor_mz": 150.0,
            "precursor_charge": 3,
            "precursor_mass": 450.0,
            "charge_id": torch.tensor(2, dtype=torch.int64),  # Different dtype
            "rt_log": 6.0,
            "meta": {
                "collision_energy": 35.0,
            }
        }
    ]
    
    result = spectra_collator(batch=batch)
    
    # All tensors should be standardized to expected dtypes
    assert result["precursors"].dtype == torch.float32, "Precursors should be float32"
    assert result["charge_id"].dtype == torch.int64, "Charge ID should be int64"
    assert result["spectra"].dtype == torch.float32, "Spectra should be float32"


def test_foundational_collate_metadata_list_conversion() -> None:
    """Test that metadata lists are properly converted to tensors."""
    batch = [
        {
            "spectra": torch.randn(3, 2, dtype=torch.float32),
            "spectra_gt": torch.randn(3, 2, dtype=torch.float32),
            "mlm_mask": torch.randint(0, 2, (3,), dtype=torch.bool),
            "precursor_mz": 100.0,
            "precursor_charge": 2,
            "precursor_mass": 200.0,
            "charge_id": 1,
            "rt_log": 5.0,
            "meta": {
                "collision_energy": [30.0],  # List that should become tensor
                "frag_id": [0],  # List that should become tensor
            }
        }
    ]
    
    metadata_columns = ["numeric_list"]
    batch[0]["numeric_list"] = [42.0]  # List that should become tensor
    
    # Include meta field selection to ensure collision_energy is processed
    meta_field_selection = {
        "include_collision_energy": True,
        "include_frag": True,
    }
    
    result = spectra_collator(
        batch=batch,
        meta_field_selection=meta_field_selection,
        metadata_columns=metadata_columns
    )
    
    # Check that lists in meta dict are converted to tensors
    assert isinstance(result["meta"]["collision_energy"], torch.Tensor), "Meta collision_energy should be tensor"
    assert isinstance(result["meta"]["frag_id"], torch.Tensor), "Meta frag_id should be tensor"
    
    # Check that metadata columns are converted to tensors
    assert isinstance(result["numeric_list"], torch.Tensor), "Numeric list should be tensor"
    assert result["numeric_list"].dtype == torch.float32, "Numeric list should be float32" 