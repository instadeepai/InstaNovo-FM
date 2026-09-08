"""Tests for split_unlabelled_data."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
import polars as pl
import pytest

from scripts.splitting.split_unlabelled_data import (
    load_existing_lsh_assignments,
    create_initial_lsh_assignments,
    assign_remaining_lsh,
    dict_to_dataframe as lsh_dict_to_dataframe,
    create_and_verify_lsh_splits,
    normalise_dataframe_schema as normalise_unlabelled_schema,
)

def _existing_df(mapping: dict) -> pl.DataFrame:
    """Build a ``(lsh_hash, split)`` DataFrame from a ``{hash: split}`` mapping."""
    return pl.DataFrame(
        {
            "lsh_hash": list(mapping.keys()),
            "split": pl.Series("split", list(mapping.values()), dtype=pl.String),
        }
    )


def _all_lsh_df(hashes: set) -> pl.DataFrame:
    """Build a single-column ``lsh_hash`` DataFrame from a set of hashes."""
    return pl.DataFrame({"lsh_hash": sorted(hashes)})


def _assignments_to_dict(df: pl.DataFrame) -> dict:
    """Collapse a ``(lsh_hash, split)`` DataFrame back to a ``{hash: split}`` dict."""
    return dict(zip(df["lsh_hash"].to_list(), df["split"].to_list()))


class TestSplitUnlabelledData:
    """Test suite for split_unlabelled_data.py - LSH-based splitting."""

    def test_create_initial_lsh_assignments_respects_existing(self) -> None:
        """Test that initial LSH assignments respect existing assignments."""
        existing_assignments = {
            100: "train",
            200: "val",
            300: "test",
        }
        all_lsh = {100, 200, 300, 400, 500}  # 3 existing + 2 new

        split_df = create_initial_lsh_assignments(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )

        # Check that existing assignments are preserved
        train_hashes = set(split_df.filter(pl.col("split") == "train")["lsh_hash"])
        val_hashes = set(split_df.filter(pl.col("split") == "val")["lsh_hash"])
        test_hashes = set(split_df.filter(pl.col("split") == "test")["lsh_hash"])

        assert 100 in train_hashes
        assert 200 in val_hashes
        assert 300 in test_hashes

        # New LSH hashes should be unassigned (None)
        unassigned = split_df.filter(pl.col("split").is_null())
        unassigned_hashes = set(unassigned["lsh_hash"])
        assert 400 in unassigned_hashes
        assert 500 in unassigned_hashes

    def test_assign_remaining_lsh_achieves_target_ratios(self) -> None:
        """Test that assign_remaining_lsh achieves approximately 80/10/10 split."""
        # Create 100 LSH hashes with 3 pre-assigned to avoid schema mismatch
        lsh_hashes = list(range(100))
        splits: list[str | None] = ["train", "val", "test"] + [None] * 97
        split_df = pl.DataFrame({"lsh_hash": lsh_hashes, "split": splits})

        result_df = assign_remaining_lsh(split_df)

        # Count split assignments
        train_count = len(result_df.filter(pl.col("split") == "train"))
        val_count = len(result_df.filter(pl.col("split") == "val"))
        test_count = len(result_df.filter(pl.col("split") == "test"))
        total = train_count + val_count + test_count

        # Should be approximately 80/10/10 (allow some rounding variance)
        assert total == 100
        assert 78 <= train_count <= 82, f"Expected ~80 train, got {train_count}"
        assert 8 <= val_count <= 12, f"Expected ~10 val, got {val_count}"
        assert 8 <= test_count <= 12, f"Expected ~10 test, got {test_count}"

    def test_no_lsh_leakage_between_splits(self) -> None:
        """Test that no LSH hash appears in multiple splits (critical leakage test)."""
        # Create LSH hashes with some pre-assigned
        existing_assignments = {i: "train" for i in range(40)}
        existing_assignments.update({i: "val" for i in range(40, 50)})
        existing_assignments.update({i: "test" for i in range(50, 60)})

        all_lsh = set(range(100))  # 60 existing + 40 new

        # Create initial assignments
        split_df = create_initial_lsh_assignments(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )

        # Assign remaining
        result_df = assign_remaining_lsh(split_df)

        # Extract hash sets for each split
        train_hashes = set(result_df.filter(pl.col("split") == "train")["lsh_hash"])
        val_hashes = set(result_df.filter(pl.col("split") == "val")["lsh_hash"])
        test_hashes = set(result_df.filter(pl.col("split") == "test")["lsh_hash"])

        # CRITICAL: No LSH hash should appear in multiple splits
        assert train_hashes.isdisjoint(
            val_hashes
        ), f"Leakage between train and val: {train_hashes & val_hashes}"
        assert train_hashes.isdisjoint(
            test_hashes
        ), f"Leakage between train and test: {train_hashes & test_hashes}"
        assert val_hashes.isdisjoint(
            test_hashes
        ), f"Leakage between val and test: {val_hashes & test_hashes}"

    def test_lsh_dict_to_dataframe_creates_correct_format(self) -> None:
        """Test that dict_to_dataframe creates correct DataFrame for output."""
        split_dict = {
            100: "train",
            200: "train",
            300: "val",
            400: "test",
            500: "test",
        }

        df = lsh_dict_to_dataframe(split_dict)

        # Check all hashes are present
        all_hashes = set(df["lsh_hash"])
        expected = {100, 200, 300, 400, 500}
        assert all_hashes == expected

        # Check splits are correct
        train_hashes = set(df.filter(pl.col("split") == "train")["lsh_hash"])
        assert train_hashes == {100, 200}

        val_hashes = set(df.filter(pl.col("split") == "val")["lsh_hash"])
        assert val_hashes == {300}

        test_hashes = set(df.filter(pl.col("split") == "test")["lsh_hash"])
        assert test_hashes == {400, 500}

    def test_create_and_verify_lsh_splits_updates_existing_assignments(self) -> None:
        """Test that create_and_verify_lsh_splits updates existing with new assignments."""
        # Initial existing assignments
        existing_assignments = {
            100: "train",
            200: "val",
            300: "test",
        }

        # New batch of LSH hashes - some overlap, some new
        all_lsh = {100, 200, 300, 400, 500, 600}  # 3 existing + 3 new

        # Call create_and_verify_lsh_splits
        split_df, updated_df = create_and_verify_lsh_splits(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )
        updated_assignments = _assignments_to_dict(updated_df)

        # Check that existing assignments are preserved
        assert updated_assignments[100] == "train"
        assert updated_assignments[200] == "val"
        assert updated_assignments[300] == "test"

        # Check that NEW hashes were added to updated_assignments
        new_hashes = {400, 500, 600}
        for lsh_hash in new_hashes:
            assert lsh_hash in updated_assignments, f"Hash {lsh_hash} not assigned"
            assert updated_assignments[lsh_hash] in ["train", "val", "test"]

        # Check that the DataFrame has all hashes from current batch
        df_hashes = set(split_df["lsh_hash"])
        assert df_hashes == all_lsh

    def test_lsh_assignments_file_contains_all_hashes(self) -> None:
        """Test that the LSH assignments output contains all hashes (historical + new)."""
        # Initial existing assignments
        existing_assignments = {
            100: "train",
            200: "val",
        }

        # Process new hashes
        all_lsh = {100, 300, 400}  # 1 existing + 2 new

        _, updated_df = create_and_verify_lsh_splits(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )

        # Simulate writing lsh_assignments.parquet (as the real script does)
        output_dir = self.data_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lsh_assignments.parquet"
        updated_df.write_parquet(str(output_path))

        # Read the parquet back and verify contents
        assert output_path.exists(), "lsh_assignments.parquet was not created"

        written_df = pl.read_parquet(output_path)

        # Check ALL hashes are in the file (historical + new)
        written_hashes = set(written_df["lsh_hash"])
        # Should include: existing (100, 200) + new assigned from all_lsh (300, 400)
        expected_hashes = set(existing_assignments.keys()) | {300, 400}
        assert (
            written_hashes == expected_hashes
        ), f"Missing hashes in file: {expected_hashes - written_hashes}"

    def test_no_leakage_after_multiple_lsh_updates(self) -> None:
        """Test that no LSH leakage occurs after multiple incremental updates."""
        # Start with some seed hashes to avoid schema mismatch bug
        existing_df = _existing_df(
            {
                0: "train",
                1: "val",
                2: "test",
            }
        )

        # Process 3 batches of LSH hashes with some overlap
        for batch_num in range(3):
            batch_hashes = {batch_num * 30 + i for i in range(30)}
            # Add some overlap with previous batches
            if batch_num > 0:
                batch_hashes |= {(batch_num - 1) * 30 + i for i in range(5)}
            # Include seed hashes in first batch
            if batch_num == 0:
                batch_hashes |= {0, 1, 2}

            _, existing_df = create_and_verify_lsh_splits(
                _all_lsh_df(batch_hashes), existing_df
            )

        # Final verification: no hash should be in multiple splits
        existing_assignments = _assignments_to_dict(existing_df)
        train_hashes = {h for h, s in existing_assignments.items() if s == "train"}
        val_hashes = {h for h, s in existing_assignments.items() if s == "val"}
        test_hashes = {h for h, s in existing_assignments.items() if s == "test"}

        assert train_hashes.isdisjoint(
            val_hashes
        ), f"Leakage between train and val: {train_hashes & val_hashes}"
        assert train_hashes.isdisjoint(
            test_hashes
        ), f"Leakage between train and test: {train_hashes & test_hashes}"
        assert val_hashes.isdisjoint(
            test_hashes
        ), f"Leakage between val and test: {val_hashes & test_hashes}"

    def test_all_lsh_hashes_assigned(self) -> None:
        """Test that all LSH hashes get assigned to a split."""
        all_lsh = set(range(50))
        # Pre-assign a few to avoid schema mismatch
        existing_assignments = {0: "train", 1: "val", 2: "test"}

        split_df = create_initial_lsh_assignments(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )
        result_df = assign_remaining_lsh(split_df)

        # All hashes should be assigned
        assigned_hashes = set(result_df["lsh_hash"])
        assert assigned_hashes == all_lsh

        # No hash should have None split
        unassigned = result_df.filter(pl.col("split").is_null())
        assert len(unassigned) == 0, f"Found unassigned hashes: {unassigned}"

    def test_existing_assignments_not_changed(self) -> None:
        """Test that existing assignments are never changed, only new ones added."""
        # Create existing assignments
        existing_assignments = {
            100: "train",
            200: "val",
            300: "test",
        }
        original_assignments = existing_assignments.copy()

        # Process new hashes that include some existing ones
        all_lsh = {100, 200, 300, 400, 500}

        _, updated_df = create_and_verify_lsh_splits(
            _all_lsh_df(all_lsh), _existing_df(existing_assignments)
        )
        updated_assignments = _assignments_to_dict(updated_df)

        # Original assignments should NOT be changed
        for lsh_hash, expected_split in original_assignments.items():
            assert updated_assignments[lsh_hash] == expected_split, (
                f"Hash {lsh_hash} was reassigned from {expected_split} "
                f"to {updated_assignments[lsh_hash]}"
            )

    def test_normalise_unlabelled_schema_adds_missing_columns(self) -> None:
        """Test that normalise_dataframe_schema adds missing columns."""
        # Create a DataFrame with some columns
        df = pl.DataFrame(
            {
                "index": [1, 2, 3],
                "precursor_mz": [500.0, 600.0, 700.0],
            }
        )

        # Define a reference schema with additional columns
        reference_schema = {
            "index": pl.Int64,
            "precursor_mz": pl.Float64,
            "retention_time": pl.Float64,
            "precursor_charge": pl.Int64,
        }

        result_df = normalise_unlabelled_schema(df, reference_schema)

        # Check all columns are present
        assert set(result_df.columns) == set(reference_schema.keys())

        # Check missing columns are null
        assert result_df["retention_time"].null_count() == 3
        assert result_df["precursor_charge"].null_count() == 3

        # Check existing columns are preserved
        assert result_df["index"].to_list() == [1, 2, 3]
        assert result_df["precursor_mz"].to_list() == [500.0, 600.0, 700.0]

    def test_load_existing_lsh_assignments_from_file(self) -> None:
        """Test loading LSH assignments from a parquet file."""
        # Create a test assignments file
        test_data = pl.DataFrame(
            {
                "lsh_hash": [100, 200, 300, 400],
                "split": ["train", "train", "val", "test"],
            }
        )
        output_path = self.data_dir / "test_lsh_assignments.parquet"
        test_data.write_parquet(str(output_path))

        # Load the assignments (returned columnar as a DataFrame)
        assignments = _assignments_to_dict(
            load_existing_lsh_assignments(str(output_path), str(self.data_dir))
        )

        # Verify assignments
        assert assignments[100] == "train"
        assert assignments[200] == "train"
        assert assignments[300] == "val"
        assert assignments[400] == "test"
        assert len(assignments) == 4

    def test_load_existing_lsh_assignments_returns_empty_if_no_file(self) -> None:
        """Test load_existing_lsh_assignments returns an empty frame if file missing."""
        assignments = load_existing_lsh_assignments(
            "/nonexistent/path.parquet", str(self.data_dir)
        )
        assert assignments.is_empty()
        assert assignments.columns == ["lsh_hash", "split"]

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"

        # Create test directories
        self.data_dir.mkdir(parents=True, exist_ok=True)

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)


class TestGetSpectra:
    """Tests for the script's spectrum conversion helper."""

    def test_get_spectra_pads_and_truncates_arrays(self) -> None:
        """Peak lists should become fixed-size float32 channel arrays."""
        from scripts.splitting.split_unlabelled_data import get_spectra
        import numpy as np

        test_data = {
            "mz_array": [
                [100.5, 200.3, 300.7, 400.1, 500.9],
                [150.2, 250.8, 350.4, 450.6, 550.3, 650.1, 750.9],
                [120.0, 240.0, 360.0],
            ],
            "intensity_array": [
                [0.5, 0.8, 0.3, 0.9, 0.2],
                [0.7, 0.4, 0.6, 0.3, 0.8, 0.1, 0.5],
                [0.9, 0.6, 0.4],
            ],
        }
        df = pl.DataFrame(test_data)

        spectra = get_spectra(df, target_len=6)

        assert spectra.shape == (3, 2, 6)
        assert spectra.dtype == np.float32
        np.testing.assert_allclose(spectra[0, 0, :5], test_data["mz_array"][0])
        assert spectra[0, 0, 5] == 0
        np.testing.assert_allclose(
            spectra[1, 0], test_data["mz_array"][1][:6]
        )
        np.testing.assert_allclose(
            spectra[2, 1, :3], test_data["intensity_array"][2]
        )
        np.testing.assert_array_equal(spectra[2, 1, 3:], 0)
