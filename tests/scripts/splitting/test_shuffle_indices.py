"""Tests for shuffle_indices."""

import polars as pl
import pytest

from scripts.splitting.shuffle_indices import (
    shuffle_split_by_indices,
    shuffle_all_splits,
    get_split_info,
    get_parquet_files,
    count_rows_in_file,
    create_row_indices,
    shuffle_indices,
    chunk_indices,
    group_indices_by_file,
    read_specific_rows,
    write_chunk_file,
)

class TestShuffleIndices:
    """Test suite for index-based shuffling."""

    def test_get_split_info(self) -> None:
        """Test getting split information."""
        split_info = get_split_info(str(self.split_dir), "train", 150)

        assert split_info.split_type == "train"
        assert split_info.total_rows == 300  # 3 files * 100 rows each
        assert split_info.chunk_size == 150
        assert split_info.num_chunks == 2  # 300 rows / 150 chunk_size = 2 chunks
        assert len(split_info.original_files) == 3
        assert len(split_info.original_row_counts) == 3

    def test_create_row_indices(self) -> None:
        """Test creating row indices."""
        split_info = get_split_info(str(self.split_dir), "train", 150)
        indices = create_row_indices(split_info)

        assert len(indices) == 300
        assert all(hasattr(idx, "file_path") for idx in indices)
        assert all(hasattr(idx, "file_id") for idx in indices)
        assert all(hasattr(idx, "row_index") for idx in indices)

    def test_shuffle_indices(self) -> None:
        """Test shuffling indices."""
        split_info = get_split_info(str(self.split_dir), "train", 150)
        indices = create_row_indices(split_info)

        # Test with seed
        shuffled1 = shuffle_indices(indices, seed=42)
        shuffled2 = shuffle_indices(indices, seed=42)

        # Same seed should produce same result
        assert shuffled1 == shuffled2

        # Different seed should produce different result
        shuffled3 = shuffle_indices(indices, seed=123)
        assert shuffled1 != shuffled3

    def test_chunk_indices(self) -> None:
        """Test chunking indices."""
        split_info = get_split_info(str(self.split_dir), "train", 150)
        indices = create_row_indices(split_info)
        shuffled = shuffle_indices(indices, seed=42)
        chunks = chunk_indices(shuffled, 150)

        assert len(chunks) == 2
        assert len(chunks[0]) == 150
        assert len(chunks[1]) == 150

    def test_group_indices_by_file(self) -> None:
        """Test grouping indices by file."""
        split_info = get_split_info(str(self.split_dir), "train", 150)
        indices = create_row_indices(split_info)
        shuffled = shuffle_indices(indices, seed=42)
        chunks = chunk_indices(shuffled, 150)
        file_groups = group_indices_by_file(chunks)

        assert len(file_groups) == 3  # 3 original files
        assert all(
            len(chunk_data) == 2 for chunk_data in file_groups.values()
        )  # 2 chunks

    def test_read_specific_rows(self) -> None:
        """Test reading specific rows from parquet file."""
        file_path = str(self.split_dir / "train_0.parquet")
        row_indices = [0, 5, 10, 15]

        df = read_specific_rows(file_path, row_indices)

        assert len(df) == 4
        assert df["id"].to_list() == [0, 5, 10, 15]

    def test_write_chunk_file(self) -> None:
        """Test writing chunk file."""
        test_data = [
            pl.DataFrame({"id": [1, 2, 3], "sequence": ["A", "B", "C"]}),
            pl.DataFrame({"id": [4, 5], "sequence": ["D", "E"]}),
        ]

        write_chunk_file(test_data, str(self.output_dir), "train", 0)

        output_file = self.output_dir / "train_0.parquet"
        assert output_file.exists()

        # Verify data
        df = pl.read_parquet(output_file)
        assert len(df) == 5
        assert df["id"].to_list() == [1, 2, 3, 4, 5]

    def test_shuffle_split_by_indices(self) -> None:
        """Test complete shuffle split by indices."""
        shuffle_split_by_indices(
            str(self.split_dir),
            "train",
            chunk_size=150,
            seed=42,
            output_dir=str(self.output_dir),
        )

        # Check output files
        output_files = list(self.output_dir.glob("train_*.parquet"))
        assert len(output_files) == 2

        # Verify total row count is preserved
        total_rows = sum(len(pl.read_parquet(f)) for f in output_files)
        assert total_rows == 300

    def test_shuffle_all_splits(self) -> None:
        """Test shuffling all splits."""
        shuffle_all_splits(
            str(self.test_dir),  # Pass parent directory containing lcfm_splits
            chunk_size=150,
            seed=42,
            output_dir=str(self.output_dir),
        )

        # Check output files for all splits
        split_output_dir = self.output_dir / "lcfm_splits"
        for split_type in ["train", "valid", "test"]:
            output_files = list(split_output_dir.glob(f"{split_type}_*.parquet"))
            assert len(output_files) == 2  # 300 rows / 150 chunk_size = 2 chunks

    def test_determinism(self) -> None:
        """Test that shuffling is deterministic with same seed."""
        # First run
        shuffle_split_by_indices(
            str(self.split_dir),
            "train",
            chunk_size=150,
            seed=42,
            output_dir=str(self.output_dir / "run1"),
        )

        # Second run with same seed
        shuffle_split_by_indices(
            str(self.split_dir),
            "train",
            chunk_size=150,
            seed=42,
            output_dir=str(self.output_dir / "run2"),
        )

        # Compare outputs
        files1 = sorted(self.output_dir.glob("run1/train_*.parquet"))
        files2 = sorted(self.output_dir.glob("run2/train_*.parquet"))

        assert len(files1) == len(files2)

        for f1, f2 in zip(files1, files2):
            df1 = pl.read_parquet(f1)
            df2 = pl.read_parquet(f2)
            assert df1.equals(df2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        # First run
        shuffle_split_by_indices(
            str(self.split_dir),
            "train",
            chunk_size=150,
            seed=42,
            output_dir=str(self.output_dir / "seed42"),
        )

        # Second run with different seed
        shuffle_split_by_indices(
            str(self.split_dir),
            "train",
            chunk_size=150,
            seed=123,
            output_dir=str(self.output_dir / "seed123"),
        )

        # Compare outputs - they should be different
        files1 = sorted(self.output_dir.glob("seed42/train_*.parquet"))
        files2 = sorted(self.output_dir.glob("seed123/train_*.parquet"))

        assert len(files1) == len(files2)

        # At least one file should be different
        different = False
        for f1, f2 in zip(files1, files2):
            df1 = pl.read_parquet(f1)
            df2 = pl.read_parquet(f2)
            if not df1.equals(df2):
                different = True
                break

        assert different, "Different seeds should produce different results"

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self, shuffle_indices_env) -> None:
        """Bind shared shuffle-indices fixtures onto the test class."""
        self.test_dir = shuffle_indices_env.test_dir
        self.split_dir = shuffle_indices_env.split_dir
        self.output_dir = shuffle_indices_env.output_dir

class TestShuffleIndicesHelpers:
    """Helpers in shuffle_indices.py that the 2-pass tests previously covered."""

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self, shuffle_2pass_env) -> None:
        self.split_dir = shuffle_2pass_env.split_dir

    def test_get_parquet_files(self) -> None:
        """Test getting parquet files."""
        files = get_parquet_files(str(self.split_dir), "train")

        assert len(files) == 5
        assert all(f.endswith(".parquet") for f in files)
        assert all("train_" in f for f in files)

    def test_count_rows_in_file(self) -> None:
        """Test counting rows in file."""
        file_path = str(self.split_dir / "train_0.parquet")
        row_count = count_rows_in_file(file_path)

        assert row_count == 200

class TestShuffleIndicesIntegration:
    """Integration coverage for index-based shuffling."""

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self, shuffle_integration_env) -> None:
        self.test_dir = shuffle_integration_env.test_dir
        self.split_dir = shuffle_integration_env.split_dir
        self.output_dir = shuffle_integration_env.output_dir

    def test_complete_shuffle_workflow_indices(self) -> None:
        """Test complete shuffle workflow using indices method."""
        # Shuffle all splits using indices method
        shuffle_all_splits(
            str(self.test_dir),  # Pass parent directory containing lcfm_splits
            chunk_size=400,
            seed=42,
            output_dir=str(self.output_dir / "indices"),
        )

        # Verify all splits were processed
        split_output_dir = self.output_dir / "indices" / "lcfm_splits"
        for split_type in ["train", "valid", "test"]:
            output_files = list(split_output_dir.glob(f"{split_type}_*.parquet"))
            assert len(output_files) > 0

            # Verify row count preservation
            total_rows = sum(len(pl.read_parquet(f)) for f in output_files)
            expected_rows = {"train": 8 * 250, "valid": 2 * 200, "test": 2 * 150}[
                split_type
            ]
            assert total_rows == expected_rows
    def test_error_handling(self) -> None:
        """Test error handling for invalid inputs."""
        # Test with non-existent directory
        with pytest.raises(ValueError, match=".*No .* files found.*"):
            shuffle_split_by_indices(
                "/non/existent/path",
                "train",
                chunk_size=150,
                seed=42,
                output_dir=str(self.output_dir),
            )

        # Test with invalid split type
        with pytest.raises(ValueError, match=".*invalid.*"):
            shuffle_split_by_indices(
                str(self.split_dir),
                "invalid_split",
                chunk_size=150,
                seed=42,
                output_dir=str(self.output_dir),
            )
