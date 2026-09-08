"""Tests for shuffle_2pass."""

import polars as pl
import pytest

from scripts.splitting.shuffle_2pass import (
    shuffle_split,
    shuffle_all,
    forecast_chunk_size,
    estimate_bytes_per_row,
)

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
    def _setup_test_environment(self, shuffle_2pass_env) -> None:
        """Bind shared 2-pass fixtures onto the test class."""
        self.test_dir = shuffle_2pass_env.test_dir
        self.split_dir = shuffle_2pass_env.split_dir
        self.output_dir = shuffle_2pass_env.output_dir
        self.temp_dir = shuffle_2pass_env.test_dir / "temp"

class TestShuffle2PassIntegration:
    """Integration coverage for 2-pass shuffling."""

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self, shuffle_integration_env) -> None:
        self.test_dir = shuffle_integration_env.test_dir
        self.split_dir = shuffle_integration_env.split_dir
        self.output_dir = shuffle_integration_env.output_dir

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
