"""Shuffle ``split_labelled_data`` output entirely in RAM.

``split_labelled_data.py`` writes shards grouped by input file, so rows arrive
correlated; training needs them in random order. When a whole split fits in
memory this is the simplest way to get a true global shuffle — one load, one
sample, re-chunked to ``--target-chunk-size``. Splits are processed one at a
time (train, then valid, then test), so the RAM requirement is the largest
single split, not the whole corpus. Use ``shuffle_2pass.py`` when even that
does not fit.

CLI::

    python scripts/splitting/shuffle_in_ram.py --help
    python scripts/splitting/shuffle_in_ram.py \
        --input-dir lcfm_splits \
        --output-dir lcfm_shuffled \
        --target-chunk-size 400000 \
        --seed 42
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Annotated, List, Optional

import polars as pl
import typer

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Shuffle split_labelled_data-style parquet shards entirely in RAM",
    no_args_is_help=True,
    add_completion=False,
)


def get_parquet_files(split_dir: str, split_type: str) -> List[str]:
    """List one split's shards in a stable order, so a seeded run is reproducible.

    Args:
        split_dir: Directory holding the ``{split}_*.parquet`` shards.
        split_type: Split prefix to match, e.g. ``train``.

    Returns:
        The matching paths, sorted; empty when the split is absent.
    """
    pattern = os.path.join(split_dir, f"{split_type}_*.parquet")
    return sorted(glob.glob(pattern))


def count_rows_in_file(file_path: str) -> int:
    """Read a row count from parquet metadata, so verification costs no full scan.

    Args:
        file_path: Parquet file to measure.

    Returns:
        The file's row count.
    """
    lazy_df = pl.scan_parquet(file_path)
    count: int = lazy_df.select(pl.len()).collect().item()
    return count


def _analyze_input_files(parquet_files: List[str]) -> tuple[int, List[int]]:
    """Record the expected row total up front so the write can be verified against it."""
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
    """Materialise the whole split at once, which is what makes a single global sample possible."""
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
    """Re-shard after shuffling so output files stay a manageable size for training."""
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
    """Globally shuffle a single split, checking nothing was lost on the way out.

    Missing splits are skipped rather than treated as an error, so the same call
    works on directories that only carry some of train/valid/test.

    Args:
        split_dir: Directory holding the split's input shards.
        split_type: Split prefix to process, e.g. ``train``.
        output_dir: Destination for the reshuffled shards.
        chunk_size: Rows per output shard.
        seed: Fixes the shuffle so the run can be reproduced.
    """
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
    """Reshuffle a whole split-output folder in one call.

    Args:
        input_dir: Directory produced by ``split_labelled_data.py``.
        output_dir: Destination for the reshuffled shards; created if absent.
        chunk_size: Rows per output shard.
        seed: Fixes the shuffle so the run can be reproduced.

    Raises:
        FileNotFoundError: If *input_dir* does not exist, caught before any
            output directory is created.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Shuffling split output folder: %s -> %s", input_dir, output_dir)
    for split_type in ("train", "valid", "test"):
        ram_shuffle_split(str(input_path), split_type, output_dir, chunk_size, seed)


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory with train_/valid_/test_ parquet shards",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for shuffled shards"),
    ],
    target_chunk_size: Annotated[
        int,
        typer.Option("--target-chunk-size", help="Rows per output shard file"),
    ],
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Globally shuffle train/valid/test parquet shards that fit in RAM."""
    configure_script_logging(verbose=verbose)

    shuffle_split_labelled_output_folder(
        str(input_dir),
        str(output_dir),
        target_chunk_size,
        seed,
    )
    logger.info("\nShuffling complete!")


if __name__ == "__main__":
    app()
