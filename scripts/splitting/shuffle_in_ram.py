"""RAM-based shuffle for split_labelled_data-style output folders.

Reads a flat directory of ``train_*.parquet``, ``valid_*.parquet``, ``test_*.parquet``
(``split_labelled_data.py`` layout), shuffles rows within each split, and writes new
shards as ``{split}_{i}.parquet`` under ``--output-dir`` using ``--target-chunk-size``.

Requires enough RAM to hold each split (train, then valid, then test) in memory.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import List, Optional

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_parquet_files(split_dir: str, split_type: str) -> List[str]:
    """Get all parquet files for a specific split type (train/valid/test)."""
    pattern = os.path.join(split_dir, f"{split_type}_*.parquet")
    return sorted(glob.glob(pattern))


def count_rows_in_file(file_path: str) -> int:
    """Count rows in a parquet file efficiently."""
    lazy_df = pl.scan_parquet(file_path)
    count: int = lazy_df.select(pl.len()).collect().item()
    return count


def _analyze_input_files(parquet_files: List[str]) -> tuple[int, List[int]]:
    """Analyze input files and return total rows and per-file row counts."""
    logger.info("Found %d input files:", len(parquet_files))
    total_rows = 0
    file_sizes: List[int] = []
    for file_path in parquet_files:
        row_count = count_rows_in_file(file_path)
        total_rows += row_count
        file_sizes.append(row_count)
        logger.info("  %s: %s rows", os.path.basename(file_path), f"{row_count:,}")

    logger.info("Total rows: %s", f"{total_rows:,}")
    return total_rows, file_sizes


def _read_all_data_to_ram(parquet_files: List[str]) -> pl.DataFrame:
    """Read all parquet files into a single DataFrame in RAM."""
    logger.info("Reading all data into RAM...")
    lazy_dfs = [pl.scan_parquet(file_path) for file_path in parquet_files]
    combined_lazy_df = pl.concat(lazy_dfs, how="vertical_relaxed")
    all_data = combined_lazy_df.collect()
    logger.info("Loaded %s rows into RAM", f"{len(all_data):,}")
    return all_data


def _shuffle_and_write_chunks(
    data: pl.DataFrame,
    output_dir: str,
    split_type: str,
    chunk_size: int,
    seed: Optional[int],
) -> int:
    """Shuffle data and write to chunks."""
    logger.info("Shuffling %s rows globally...", f"{len(data):,}")
    shuffled_data = data.sample(
        n=len(data), with_replacement=False, seed=seed, shuffle=True
    )

    logger.info("Writing shuffled data to chunks of size %s...", f"{chunk_size:,}")
    total_written = 0
    chunk_id = 0

    for start_idx in range(0, len(shuffled_data), chunk_size):
        end_idx = min(start_idx + chunk_size, len(shuffled_data))
        chunk_data = shuffled_data.slice(start_idx, end_idx - start_idx)
        output_file = os.path.join(output_dir, f"{split_type}_{chunk_id}.parquet")
        chunk_data.write_parquet(output_file)
        logger.info(
            "  Wrote %s rows to %s",
            f"{len(chunk_data):,}",
            os.path.basename(output_file),
        )
        total_written += len(chunk_data)
        chunk_id += 1

    logger.info("Created %d output chunks", chunk_id)
    logger.info("Total output rows: %s", f"{total_written:,}")
    return total_written


def ram_shuffle_split(
    split_dir: str,
    split_type: str,
    output_dir: str,
    chunk_size: int,
    seed: Optional[int] = None,
) -> None:
    """Shuffle one split (train / valid / test) and write shards under output_dir."""
    logger.info(
        "Processing %s split in %s (chunk size %s)",
        split_type,
        split_dir,
        f"{chunk_size:,}",
    )

    parquet_files = get_parquet_files(split_dir, split_type)
    if not parquet_files:
        logger.info("No %s files found in %s", split_type, split_dir)
        return

    total_rows, _ = _analyze_input_files(parquet_files)
    os.makedirs(output_dir, exist_ok=True)
    all_data = _read_all_data_to_ram(parquet_files)
    total_output_rows = _shuffle_and_write_chunks(
        all_data, output_dir, split_type, chunk_size, seed
    )

    if total_output_rows == total_rows:
        logger.info("Row counts match for %s.", split_type)
    else:
        logger.error(
            "Row count mismatch for %s! Expected %s, got %s",
            split_type,
            f"{total_rows:,}",
            f"{total_output_rows:,}",
        )


def shuffle_split_labelled_output_folder(
    input_dir: str,
    output_dir: str,
    chunk_size: int,
    seed: Optional[int] = None,
) -> None:
    """Shuffle train/valid/test shards from input_dir into output_dir."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Shuffling split output folder: %s -> %s", input_dir, output_dir)
    for split_type in ("train", "valid", "test"):
        ram_shuffle_split(str(input_path), split_type, output_dir, chunk_size, seed)


def main() -> None:
    """Main function to parse arguments and shuffle splits."""
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle in RAM for split_labelled_data-style train_/valid_/test_ parquet shards "
            "into a new directory with the same naming pattern."
        )
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing train_*.parquet, valid_*.parquet, test_*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for shuffled train_*.parquet, valid_*.parquet, test_*.parquet",
    )
    parser.add_argument(
        "--target-chunk-size",
        "-r",
        type=int,
        required=True,
        help="Number of rows per output shard file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    shuffle_split_labelled_output_folder(
        args.input_dir,
        args.output_dir,
        args.target_chunk_size,
        args.seed,
    )
    logger.info("\nShuffling complete!")


if __name__ == "__main__":
    main()
