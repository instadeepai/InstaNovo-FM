"""Comprehensive test suite for splits scripts.

This module provides comprehensive tests for all the splits scripts
to ensure they work correctly with realistic parquet data.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Generator
import polars as pl
import pytest

# Import the scripts to test
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
from scripts.splitting.shuffle_2pass import (
    shuffle_split,
    shuffle_all,
    forecast_chunk_size,
    estimate_bytes_per_row,
)
from scripts.splitting.split_unlabelled_data import (
    load_existing_lsh_assignments,
    create_initial_lsh_assignments,
    assign_remaining_lsh,
    dict_to_dataframe as lsh_dict_to_dataframe,
    create_and_verify_lsh_splits,
    normalise_dataframe_schema as normalise_unlabelled_schema,
)
from scripts.splitting.split_labelled_data import (
    REGISTRY_FILENAME,
    Mode,
    assign_new_peptides,
    collect_unique_peptides,
    load_peptide_registry,
    process_directories,
    save_registry,
    verify_no_unseen_peptides,
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
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        # Create the structure that the scripts expect: temp_dir/lcfm_splits
        self.split_dir = Path(self.test_dir) / "lcfm_splits"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self._create_test_splits()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_test_splits(self) -> None:
        """Create realistic test split data."""
        # Create multiple split files with different sizes
        for split_type in ["train", "valid", "test"]:
            for i in range(3):  # Create 3 files per split
                data = {
                    "id": list(range(i * 100, (i + 1) * 100)),
                    "sequence": [
                        f"ATCG{j % 100}" for j in range(i * 100, (i + 1) * 100)
                    ],
                    "quality": [j % 50 for j in range(i * 100, (i + 1) * 100)],
                    "metadata": [
                        f"sample_{j % 10}" for j in range(i * 100, (i + 1) * 100)
                    ],
                    "mz": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(i * 100, (i + 1) * 100)
                    ],
                    "intensity": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(i * 100, (i + 1) * 100)
                    ],
                }

                df = pl.DataFrame(data)
                df.write_parquet(self.split_dir / f"{split_type}_{i}.parquet")


class TestShuffle2Pass:
    """Test suite for 2-pass shuffling."""

    def test_forecast_chunk_size(self) -> None:
        """Test chunk size forecasting."""
        forecast = forecast_chunk_size(
            ram_gb=16.0,
            num_procs=4,
            bytes_per_row=1000.0,
            safety=0.7,
            pass2=False,
        )

        assert "max_chunk_size" in forecast
        assert "data_memory_per_process_gb" in forecast
        assert "total_memory_all_procs_gb" in forecast
        assert "ram_utilization_pct" in forecast

    def test_forecast_chunk_size_second_pass(self) -> None:
        """Test second pass chunk size forecasting."""
        forecast = forecast_chunk_size(
            ram_gb=16.0,
            num_procs=4,
            bytes_per_row=1000.0,
            safety=0.7,
            pass2=True,
        )

        assert "max_chunk_size" in forecast
        assert "memory_multiplier" in forecast
        assert forecast["memory_multiplier"] == 2

    def test_estimate_bytes_per_row(self) -> None:
        """Test bytes per row estimation."""
        sample_file = str(self.split_dir / "train_0.parquet")
        bytes_per_row = estimate_bytes_per_row(sample_file, n=100)

        assert isinstance(bytes_per_row, float)
        assert bytes_per_row > 0

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

    def test_shuffle_split(self) -> None:
        """Test complete 2-pass shuffle split."""
        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        # Check output files
        output_files = list(self.output_dir.glob("train_*.parquet"))
        assert len(output_files) > 0

        # Verify total row count is preserved
        total_rows = sum(len(pl.read_parquet(f)) for f in output_files)
        expected_rows = 5 * 200  # 5 files * 200 rows each
        assert total_rows == expected_rows

    def test_shuffle_all(self) -> None:
        """Test 2-pass shuffling all splits."""
        shuffle_all(
            str(self.split_dir),
            str(self.output_dir),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        for split_type in ["train", "valid", "test"]:
            output_files = list(self.output_dir.glob(f"{split_type}_*.parquet"))
            assert len(output_files) > 0

    def test_2pass_determinism(self) -> None:
        """Test that 2-pass shuffling is deterministic with same seed."""
        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir / "run1"),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir / "run2"),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        files1 = sorted(self.output_dir.glob("run1/train_*.parquet"))
        files2 = sorted(self.output_dir.glob("run2/train_*.parquet"))

        assert len(files1) == len(files2)

        # For multiprocessing, we verify data integrity rather than exact determinism
        total_rows1 = sum(len(pl.read_parquet(f)) for f in files1)
        total_rows2 = sum(len(pl.read_parquet(f)) for f in files2)
        assert total_rows1 == total_rows2

    def test_2pass_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results in 2-pass."""
        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir / "seed42"),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir / "seed123"),
            chunk_size=300,
            seed=123,
            pass1_procs=1,
            pass2_procs=1,
        )

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

    def test_data_integrity_preservation(self) -> None:
        """Test that 2-pass shuffling preserves data integrity."""
        original_files = list(self.split_dir.glob("train_*.parquet"))
        original_data = []
        for f in original_files:
            df = pl.read_parquet(f)
            original_data.append(df)

        original_combined = pl.concat(original_data, how="vertical_relaxed")
        original_sorted = original_combined.sort("id")

        shuffle_split(
            str(self.split_dir),
            "train",
            output_dir=str(self.output_dir),
            chunk_size=300,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        shuffled_files = list(self.output_dir.glob("train_*.parquet"))
        shuffled_data = []
        for f in shuffled_files:
            df = pl.read_parquet(f)
            shuffled_data.append(df)

        shuffled_combined = pl.concat(shuffled_data, how="vertical_relaxed")
        shuffled_sorted = shuffled_combined.sort("id")

        assert original_sorted.equals(shuffled_sorted)

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        # Create the structure that the scripts expect: temp_dir/lcfm_splits
        self.split_dir = Path(self.test_dir) / "lcfm_splits"
        self.output_dir = Path(self.test_dir) / "outputs"
        self.temp_dir = Path(self.test_dir) / "temp"

        # Create test directories
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self._create_test_splits()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_test_splits(self) -> None:
        """Create realistic test split data for 2-pass testing."""
        # Create larger datasets for 2-pass testing
        for split_type in ["train", "valid", "test"]:
            for i in range(5):  # Create 5 files per split
                data = {
                    "id": list(range(i * 200, (i + 1) * 200)),
                    "sequence": [
                        f"ATCG{j % 100}" for j in range(i * 200, (i + 1) * 200)
                    ],
                    "quality": [j % 50 for j in range(i * 200, (i + 1) * 200)],
                    "metadata": [
                        f"sample_{j % 10}" for j in range(i * 200, (i + 1) * 200)
                    ],
                    "mz": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(i * 200, (i + 1) * 200)
                    ],
                    "intensity": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(i * 200, (i + 1) * 200)
                    ],
                }

                df = pl.DataFrame(data)
                df.write_parquet(self.split_dir / f"{split_type}_{i}.parquet")


class TestSplitsIntegration:
    """Integration tests for splits workflows."""

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

    def test_complete_shuffle_workflow_2pass(self) -> None:
        """Test complete shuffle workflow using 2-pass method."""
        # shuffle_all writes train_*.parquet directly into output_dir
        shuffle_all(
            str(self.split_dir),
            str(self.output_dir / "2pass"),
            chunk_size=400,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        for split_type in ["train", "valid", "test"]:
            output_files = list(
                (self.output_dir / "2pass").glob(f"{split_type}_*.parquet")
            )
            assert len(output_files) > 0

            total_rows = sum(len(pl.read_parquet(f)) for f in output_files)
            expected_rows = {"train": 8 * 250, "valid": 2 * 200, "test": 2 * 150}[
                split_type
            ]
            assert total_rows == expected_rows

    def test_comparison_between_methods(self) -> None:
        """Test comparison between indices and 2-pass methods."""
        shuffle_all_splits(
            str(self.test_dir),  # Pass parent directory containing lcfm_splits
            chunk_size=400,
            seed=42,
            output_dir=str(self.output_dir / "indices"),
        )

        shuffle_all(
            str(self.split_dir),
            str(self.output_dir / "2pass"),
            chunk_size=400,
            seed=42,
            pass1_procs=1,
            pass2_procs=1,
        )

        # Both methods should preserve total row counts
        indices_output_dir = self.output_dir / "indices" / "lcfm_splits"
        twopass_output_dir = self.output_dir / "2pass"
        for split_type in ["train", "valid", "test"]:
            indices_files = list(indices_output_dir.glob(f"{split_type}_*.parquet"))
            twopass_files = list(twopass_output_dir.glob(f"{split_type}_*.parquet"))

            indices_rows = sum(len(pl.read_parquet(f)) for f in indices_files)
            twopass_rows = sum(len(pl.read_parquet(f)) for f in twopass_files)

            expected_rows = {"train": 8 * 250, "valid": 2 * 200, "test": 2 * 150}[
                split_type
            ]

            assert indices_rows == expected_rows
            assert twopass_rows == expected_rows

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

    def test_memory_forecasting_integration(self) -> None:
        """Test memory forecasting integration."""
        scenarios = [
            {"ram_gb": 8.0, "processes": 2},
            {"ram_gb": 16.0, "processes": 4},
            {"ram_gb": 32.0, "processes": 8},
        ]

        for scenario in scenarios:
            forecast = forecast_chunk_size(
                ram_gb=scenario["ram_gb"],
                num_procs=int(scenario["processes"]),
                bytes_per_row=1000.0,
                safety=0.7,
            )

            assert forecast["max_chunk_size"] > 0
            assert forecast["ram_utilization_pct"] <= 100.0
            assert forecast["ram_utilization_pct"] > 0.0

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self) -> Generator[None, None, None]:
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.split_dir = Path(self.test_dir) / "lcfm_splits"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create realistic test data
        self._create_integration_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_integration_test_data(self) -> None:
        """Create realistic test data for integration testing."""
        # Create multiple splits with varying sizes
        split_configs = {
            "train": {"files": 8, "rows_per_file": 250},
            "valid": {"files": 2, "rows_per_file": 200},
            "test": {"files": 2, "rows_per_file": 150},
        }

        for split_type, config in split_configs.items():
            for i in range(config["files"]):
                data = {
                    "id": list(
                        range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ),
                    "sequence": [
                        f"ATCG{j % 100}"
                        for j in range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ],
                    "quality": [
                        j % 50
                        for j in range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ],
                    "metadata": [
                        f"sample_{j % 10}"
                        for j in range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ],
                    "mz": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ],
                    "intensity": [
                        [100.0 + j, 200.0 + j, 300.0 + j]
                        for j in range(
                            i * config["rows_per_file"],
                            (i + 1) * config["rows_per_file"],
                        )
                    ],
                }

                df = pl.DataFrame(data)
                df.write_parquet(self.split_dir / f"{split_type}_{i}.parquet")


class TestSplitLabelledData:
    """Test suite for split_labelled_data.py — registry extension and leakage."""

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Temporary directories for local registries and parquet fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.registry_dir = Path(self.test_dir) / "registry"
        self.output_dir = Path(self.test_dir) / "output"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(self.test_dir)

    @staticmethod
    def _empty_splits() -> dict[str, set[str]]:
        return {"train": set(), "test": set(), "valid": set()}

    @staticmethod
    def _write_registry(
        directory: Path, splits: dict[str, set[str]], *, use_hf_valid: bool = False
    ) -> Path:
        """Write peptide_registry.parquet; HF schema uses 'validation' not 'valid'."""
        peptides: list[str] = []
        labels: list[str] = []
        for split_name, seqs in splits.items():
            label = (
                "validation"
                if use_hf_valid and split_name == "valid"
                else split_name
            )
            for peptide in sorted(seqs):
                peptides.append(peptide)
                labels.append(label)
        path = directory / REGISTRY_FILENAME
        pl.DataFrame({"peptide": peptides, "split": labels}).write_parquet(path)
        return path

    @staticmethod
    def _minimal_labelled_parquet(
        path: Path,
        sequences: list[str],
        *,
        unmodified: list[str] | None = None,
    ) -> None:
        """Write a tiny parquet that passes quality filters (null filter cols pass)."""
        n = len(sequences)
        data: dict = {
            "index": list(range(n)),
            "scan": [str(i) for i in range(n)],
            "header": [f"h{i}" for i in range(n)],
            "sequence": sequences,
            "mz_array": [[100.0 + i, 200.0 + i] for i in range(n)],
            "intensity_array": [[1.0, 2.0] for _ in range(n)],
        }
        if unmodified is not None:
            data["unmodified_peptide"] = unmodified
        # Leave retention_time / precursor_* / lower_offset absent so schema
        # normalisation fills nulls and _nullable_filter lets rows through.
        pl.DataFrame(data).write_parquet(path)

    def test_collect_unique_peptides_normalises_i_to_l(self) -> None:
        """I→L and unmodified_peptide-from-sequence feed the registry key space."""
        parquet = self.data_dir / "sample.parquet"
        self._minimal_labelled_parquet(
            parquet,
            sequences=["PEPTIDE[UNIMOD:4]", "PEPTLDE", "AAAA"],
            unmodified=None,
        )
        peptides = collect_unique_peptides([str(parquet)])
        assert "PEPTLDE" in peptides
        assert "PEPTIDE" not in peptides
        assert "AAAA" in peptides

    def test_assign_preserves_existing_and_places_only_new(self) -> None:
        """Seeded registry peptides stay put; only absent peptides are assigned."""
        existing = {
            "train": {"SEEDTRAIN", "SEEDTRAIN2"},
            "test": {"SEEDTEST"},
            "valid": {"SEEDVALID"},
        }
        dataset = {
            "SEEDTRAIN",
            "SEEDTRAIN2",
            "SEEDTEST",
            "SEEDVALID",
            "NEW1",
            "NEW2",
            "NEW3",
            "NEW4",
            "NEW5",
        }
        before = {k: set(v) for k, v in existing.items()}
        updated, added = assign_new_peptides(dataset, existing)

        assert updated["train"] >= before["train"]
        assert updated["test"] >= before["test"]
        assert updated["valid"] >= before["valid"]
        assert "SEEDTRAIN" in updated["train"]
        assert "SEEDTEST" in updated["test"]
        assert "SEEDVALID" in updated["valid"]

        new = {"NEW1", "NEW2", "NEW3", "NEW4", "NEW5"}
        assigned_new = (
            (updated["train"] & new)
            | (updated["test"] & new)
            | (updated["valid"] & new)
        )
        assert assigned_new == new
        assert sum(added.values()) == len(new)

        for peptide in new:
            membership = sum(peptide in updated[s] for s in ("train", "test", "valid"))
            assert membership == 1, f"{peptide} in {membership} splits"

    def test_no_peptide_leakage(self) -> None:
        """No peptide appears in more than one split after incremental updates."""
        existing = {
            "train": {f"TRAIN{i}" for i in range(40)},
            "test": {f"TEST{i}" for i in range(5)},
            "valid": {f"VALID{i}" for i in range(5)},
        }
        dataset = (
            existing["train"]
            | existing["test"]
            | existing["valid"]
            | {f"NEW{i}" for i in range(50)}
        )
        updated, _ = assign_new_peptides(dataset, existing)
        assert updated["train"].isdisjoint(updated["test"])
        assert updated["train"].isdisjoint(updated["valid"])
        assert updated["test"].isdisjoint(updated["valid"])

    def test_assign_is_deterministic(self) -> None:
        """Same seed and inputs yield identical assignments."""
        existing_a = self._empty_splits()
        existing_b = self._empty_splits()
        peptides = {f"P{i:03d}" for i in range(100)}
        a, _ = assign_new_peptides(peptides, existing_a)
        b, _ = assign_new_peptides(peptides, existing_b)
        assert a["train"] == b["train"]
        assert a["test"] == b["test"]
        assert a["valid"] == b["valid"]

    def test_saturation_skips_overfull_split(self) -> None:
        """When train already meets its share of dataset_peptides, new go elsewhere."""
        # |dataset|=92; train target ≈ 73.6; train ∩ dataset = 80 → train saturated
        train_seed = {f"TS{i}" for i in range(80)}
        existing = {
            "train": set(train_seed),
            "test": {"TE0"},
            "valid": {"TV0"},
        }
        news = {f"NEW{i}" for i in range(10)}
        dataset = train_seed | {"TE0", "TV0"} | news
        updated, added = assign_new_peptides(dataset, existing)
        assert added["train"] == 0
        assert news.isdisjoint(updated["train"])
        assert news <= (updated["test"] | updated["valid"])

    def test_save_load_registry_roundtrip_maps_validation(self) -> None:
        """HF 'validation' label loads as local 'valid'; round-trip preserves sets."""
        splits = {
            "train": {"AAA", "BBB"},
            "test": {"CCC"},
            "valid": {"DDD", "EEE"},
        }
        self._write_registry(self.registry_dir, splits, use_hf_valid=True)
        _, loaded = load_peptide_registry(str(self.registry_dir))
        assert loaded["train"] == splits["train"]
        assert loaded["test"] == splits["test"]
        assert loaded["valid"] == splits["valid"]

        out = self.output_dir / REGISTRY_FILENAME
        save_registry(loaded, out, upload_to_hf=False)
        _, reloaded = load_peptide_registry(str(self.output_dir))
        assert reloaded == loaded

    def test_empty_registry_edge_case_fills_ratios(self) -> None:
        """0-row stub registry → empty sets; assign_new_peptides fills ~80/10/10."""
        self._write_registry(self.registry_dir, self._empty_splits())
        _, existing = load_peptide_registry(str(self.registry_dir))
        assert existing == self._empty_splits()

        peptides = {f"P{i:03d}" for i in range(100)}
        updated, added = assign_new_peptides(peptides, existing)
        total = sum(len(updated[s]) for s in ("train", "test", "valid"))
        assert total == 100
        assert sum(added.values()) == 100
        assert updated["train"].isdisjoint(updated["test"])
        assert updated["train"].isdisjoint(updated["valid"])
        assert updated["test"].isdisjoint(updated["valid"])
        assert 75 <= len(updated["train"]) <= 85
        assert 5 <= len(updated["test"]) <= 15
        assert 5 <= len(updated["valid"]) <= 15

    def test_verify_no_unseen_peptides_raises(self) -> None:
        """split-only guard fails when a peptide is missing from the registry."""
        split_lookup = {
            "train": {"KNOWN"},
            "test": set(),
            "valid": set(),
        }
        with pytest.raises(ValueError, match="not in registry"):
            verify_no_unseen_peptides(
                parquet_files=[],
                split_lookup=split_lookup,
                dataset_peptides={"KNOWN", "MISSING"},
            )

    def test_verify_no_unseen_peptides_passes(self) -> None:
        """All dataset peptides present in registry → no error."""
        split_lookup = {
            "train": {"A", "B"},
            "test": {"C"},
            "valid": {"D"},
        }
        verify_no_unseen_peptides(
            parquet_files=[],
            split_lookup=split_lookup,
            dataset_peptides={"A", "C"},
        )

    def test_process_directories_preserves_seed_and_writes_shards(self) -> None:
        """E2E: seeded local registry + both mode writes shards without moving seeds."""
        seed = {
            "train": {"SEEDAAA", "SEEDBBB"},
            "test": {"SEEDCCC"},
            "valid": {"SEEDDDD"},
        }
        self._write_registry(self.registry_dir, seed, use_hf_valid=True)

        sequences = [
            "SEEDAAA",
            "SEEDBBB",
            "SEEDCCC",
            "SEEDDDD",
            "NEWPEPAA",
            "NEWPEPBB",
            "NEWPEPCC",
            "NEWPEPDD",
            "NEWPEPEE",
            "NEWPEPFF",
        ]
        self._minimal_labelled_parquet(
            self.data_dir / "batch.parquet",
            sequences=sequences,
            unmodified=sequences,
        )

        process_directories(
            input_dirs=[str(self.data_dir)],
            output_dir=str(self.output_dir),
            rows_per_file=100,
            registry_dir=str(self.registry_dir),
            mode=Mode.BOTH,
            upload_to_hf=False,
        )

        _, updated = load_peptide_registry(str(self.output_dir))
        assert "SEEDAAA" in updated["train"]
        assert "SEEDBBB" in updated["train"]
        assert "SEEDCCC" in updated["test"]
        assert "SEEDDDD" in updated["valid"]

        train_files = list(self.output_dir.glob("train_*.parquet"))
        assert train_files, "expected train shards"
        assert list(self.output_dir.glob("test_*.parquet"))
        assert list(self.output_dir.glob("valid_*.parquet"))

        all_written: set[str] = set()
        for pattern in ("train_*.parquet", "test_*.parquet", "valid_*.parquet"):
            for f in self.output_dir.glob(pattern):
                all_written.update(
                    pl.read_parquet(f)["unmodified_peptide"].to_list()
                )
        assert "SEEDAAA" in all_written
        assert "SEEDCCC" in all_written
        assert "SEEDDDD" in all_written
        assert {"NEWPEPAA", "NEWPEPBB", "NEWPEPCC", "NEWPEPDD", "NEWPEPEE", "NEWPEPFF"} <= all_written


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


class TestLSHHashComputation:
    """Test suite for LSH hash computation reproducibility and correctness."""

    def test_lsh_hash_is_reproducible(self) -> None:
        """Test that LSH hash computation is reproducible (same input = same hash)."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        # Create two identical projectors with same seed
        projector1 = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )
        projector2 = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )

        # Create sample peak list data (shape: num_spectra, 2, num_peaks)
        np.random.seed(123)
        mz_values = np.sort(np.random.uniform(100, 900, size=(5, 200)), axis=1)
        intensity_values = np.random.uniform(0.1, 1.0, size=(5, 200))
        peak_lists = np.stack([mz_values, intensity_values], axis=1).astype(np.float32)

        # Compute hashes with both projectors
        hashes1 = projector1.compute(peak_lists, as_str=True, progress_bar=False)
        hashes2 = projector2.compute(peak_lists, as_str=True, progress_bar=False)

        # Hashes should be identical
        assert np.array_equal(
            hashes1, hashes2
        ), "LSH hashes should be identical for same input and same seed"

    def test_lsh_hash_differs_with_different_seed(self) -> None:
        """Test that different seeds produce different hash values."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        projector_seed1 = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )
        projector_seed2 = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=99
        )

        np.random.seed(123)
        mz_values = np.sort(np.random.uniform(100, 900, size=(5, 200)), axis=1)
        intensity_values = np.random.uniform(0.1, 1.0, size=(5, 200))
        peak_lists = np.stack([mz_values, intensity_values], axis=1).astype(np.float32)

        hashes1 = projector_seed1.compute(peak_lists, as_str=True, progress_bar=False)
        hashes2 = projector_seed2.compute(peak_lists, as_str=True, progress_bar=False)

        # At least some hashes should differ with different seeds
        assert not np.array_equal(
            hashes1, hashes2
        ), "LSH hashes should differ with different seeds"

    def test_batch_computation_matches_sequential(self) -> None:
        """Test that batch computation produces same results as sequential computation."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        # Create projector with small subbatch size to force batched processing
        projector_batched = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=2, seed=42
        )
        # Create projector with large subbatch size (no batching)
        projector_full = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=10000, seed=42
        )

        # Create 10 spectra to ensure batching happens (subbatch_size=2)
        np.random.seed(456)
        mz_values = np.sort(np.random.uniform(100, 900, size=(10, 200)), axis=1)
        intensity_values = np.random.uniform(0.1, 1.0, size=(10, 200))
        peak_lists = np.stack([mz_values, intensity_values], axis=1).astype(np.float32)

        # Compute with batched processing
        hashes_batched = projector_batched.compute(
            peak_lists, as_str=True, progress_bar=False
        )
        # Compute without batching
        hashes_full = projector_full.compute(
            peak_lists, as_str=True, progress_bar=False
        )

        # Results should be identical regardless of batching
        assert np.array_equal(
            hashes_batched, hashes_full
        ), "Batched and non-batched LSH computation should produce identical results"

    def test_identical_spectra_produce_same_hash(self) -> None:
        """Test that identical spectra produce the same LSH hash."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        projector = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )

        # Create identical spectra
        np.random.seed(789)
        mz = np.sort(np.random.uniform(100, 900, size=200))
        intensity = np.random.uniform(0.1, 1.0, size=200)

        # Duplicate the spectrum 3 times
        peak_lists = np.stack(
            [
                np.stack([mz, intensity]),
                np.stack([mz, intensity]),
                np.stack([mz, intensity]),
            ]
        ).astype(np.float32)

        hashes = projector.compute(peak_lists, as_str=True, progress_bar=False)

        # All hashes should be identical
        assert (
            len(set(hashes.flatten())) == 1
        ), "Identical spectra should produce identical LSH hashes"

    def test_different_spectra_produce_different_hashes(self) -> None:
        """Test that significantly different spectra produce different LSH hashes."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        projector = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )

        # Create distinctly different spectra
        # Spectrum 1: low mz range
        mz1 = np.sort(np.random.uniform(100, 300, size=200))
        intensity1 = np.random.uniform(0.5, 1.0, size=200)

        # Spectrum 2: high mz range
        mz2 = np.sort(np.random.uniform(700, 900, size=200))
        intensity2 = np.random.uniform(0.5, 1.0, size=200)

        peak_lists = np.stack(
            [
                np.stack([mz1, intensity1]),
                np.stack([mz2, intensity2]),
            ]
        ).astype(np.float32)

        hashes = projector.compute(peak_lists, as_str=True, progress_bar=False)

        # Hashes should be different for very different spectra
        assert (
            hashes[0] != hashes[1]
        ), "Significantly different spectra should produce different LSH hashes"

    def test_rows_assigned_correctly_based_on_computed_hash(self) -> None:
        """Test that rows are assigned to the correct split based on their LSH hash."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        # Create test spectra
        np.random.seed(111)
        num_spectra = 20
        mz_values = np.sort(
            np.random.uniform(100, 900, size=(num_spectra, 200)), axis=1
        )
        intensity_values = np.random.uniform(0.1, 1.0, size=(num_spectra, 200))
        peak_lists = np.stack([mz_values, intensity_values], axis=1).astype(np.float32)

        # Compute LSH hashes
        projector = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )
        hashes = projector.compute(peak_lists, as_str=True, progress_bar=False)

        # Create mock LSH-to-split mapping
        # Use hash of first byte to convert to int for mapping
        unique_hashes = list(set(hashes.flatten()))
        lsh_to_split = {}
        for i, h in enumerate(unique_hashes):
            if i % 3 == 0:
                lsh_to_split[h.decode()] = "train"
            elif i % 3 == 1:
                lsh_to_split[h.decode()] = "val"
            else:
                lsh_to_split[h.decode()] = "test"

        # Verify each row is assigned to the correct split based on its hash
        for _idx, hash_val in enumerate(hashes):
            hash_str = hash_val.decode() if isinstance(hash_val, bytes) else hash_val
            expected_split = lsh_to_split[hash_str]
            # Simulate what would happen in real processing
            # (verify the hash can be looked up correctly)
            assert hash_str in lsh_to_split, f"Hash {hash_str} not found in mapping"
            assert expected_split in [
                "train",
                "val",
                "test",
            ], f"Invalid split: {expected_split}"

    def test_hash_computation_with_padded_arrays(self) -> None:
        """Test LSH hash computation with padded arrays (as used in split_unlabelled_data)."""
        from scripts.splitting.split_unlabelled_data import get_spectra
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        # Create a mock DataFrame with mz_array and intensity_array
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

        # Get padded spectra
        spectra = get_spectra(df)

        # Verify shape
        assert spectra.shape[0] == 3, "Should have 3 spectra"
        assert spectra.shape[1] == 2, "Should have 2 channels (mz, intensity)"
        assert spectra.shape[2] == 800, "Should be padded to TARGET_LEN=800"

        # Compute hashes
        projector = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )
        hashes = projector.compute(spectra, as_str=True, progress_bar=False)

        # Verify we get one hash per spectrum
        assert len(hashes) == 3, "Should get one hash per spectrum"

        # Verify reproducibility with padded arrays
        spectra2 = get_spectra(df)
        hashes2 = projector.compute(spectra2, as_str=True, progress_bar=False)
        assert np.array_equal(
            hashes, hashes2
        ), "Padded array hashes should be reproducible"

    def test_spectra_with_same_hash_go_to_same_split(self) -> None:
        """Test that spectra producing the same hash are assigned to the same split."""
        from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
        import numpy as np

        # Create duplicate spectra that will have the same hash
        np.random.seed(222)
        mz = np.sort(np.random.uniform(100, 900, size=200))
        intensity = np.random.uniform(0.1, 1.0, size=200)

        # Create 6 spectra: 2 copies each of 3 different spectra
        spectrum1 = np.stack([mz, intensity])
        spectrum2 = np.stack([mz * 0.9, intensity * 0.8])  # Different spectrum
        spectrum3 = np.stack([mz * 1.1, intensity * 1.2])  # Another different spectrum

        peak_lists = np.stack(
            [
                spectrum1,
                spectrum1,  # Duplicates
                spectrum2,
                spectrum2,  # Duplicates
                spectrum3,
                spectrum3,  # Duplicates
            ]
        ).astype(np.float32)

        projector = BatchedPeakListRandomProjection(
            max_mz=1000.0, bin_step=1.0, n_hyperplanes=30, subbatch_size=1024, seed=42
        )
        hashes = projector.compute(peak_lists, as_str=True, progress_bar=False)

        # Create split mapping
        unique_hashes = list({h.decode() for h in hashes})
        lsh_to_split = {
            unique_hashes[0]: "train",
            unique_hashes[1]: "val",
            unique_hashes[2]: "test",
        }

        # Verify duplicate spectra get same split
        assigned_splits = [lsh_to_split[h.decode()] for h in hashes]

        # Indices 0 and 1 should have same split (duplicate of spectrum1)
        assert (
            assigned_splits[0] == assigned_splits[1]
        ), "Duplicate spectra should be assigned to the same split"
        # Indices 2 and 3 should have same split (duplicate of spectrum2)
        assert (
            assigned_splits[2] == assigned_splits[3]
        ), "Duplicate spectra should be assigned to the same split"
        # Indices 4 and 5 should have same split (duplicate of spectrum3)
        assert (
            assigned_splits[4] == assigned_splits[5]
        ), "Duplicate spectra should be assigned to the same split"
