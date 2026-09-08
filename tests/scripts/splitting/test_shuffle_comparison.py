"""Cross-script comparison of shuffle_indices and shuffle_2pass."""

import polars as pl
import pytest

from scripts.splitting.shuffle_2pass import shuffle_all
from scripts.splitting.shuffle_indices import shuffle_all_splits


class TestShuffleComparison:
    """Compare indices and 2-pass shuffle methods on the same data."""

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self, shuffle_integration_env) -> None:
        self.test_dir = shuffle_integration_env.test_dir
        self.split_dir = shuffle_integration_env.split_dir
        self.output_dir = shuffle_integration_env.output_dir

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
