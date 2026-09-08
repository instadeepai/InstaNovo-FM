"""Shuffle parquet splits by shuffling row addresses rather than rows.

Only the ``(file, row)`` index list is held in memory; the actual spectra are
pulled from the original files as each output chunk is assembled. That keeps
peak memory tied to the index list instead of the data, at the cost of
re-reading source files once per chunk they contribute to. The steps are:

1. Count rows in each chunk file
2. Assign unique indices to each row (file_id, row_index)
3. Shuffle the index list
4. Create new chunks by reading specific rows from original files
5. Verify row counts match original

This is the slowest of the three shuffle scripts — prefer ``shuffle_in_ram.py``
when a split fits in memory, or ``shuffle_2pass.py`` when it does not.

CLI::

    uv run python -m scripts.splitting.shuffle_indices --help
    uv run python -m scripts.splitting.shuffle_indices \
        --input-dir /data/root \
        --output-dir shuffled_splits \
        --chunk-size 400000 \
        --seed 42
"""

import glob
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)


@dataclass
class RowIndex:
    """Addresses one row without holding it, so the shuffle can run over indices alone.

    Args:
        file_path: Source parquet file the row lives in.
        file_id: Position of that file in the split's sorted file list.
        row_index: Zero-based row offset within the file.
    """

    file_path: str
    file_id: int
    row_index: int


@dataclass
class SplitInfo:
    """Everything the shuffle needs to know about a split before touching its data.

    Gathering counts and the chunk plan once means the later passes never
    re-scan the inputs just to work out sizes.

    Args:
        split_type: Split prefix, e.g. ``train``.
        total_rows: Rows across all of the split's files; the figure output is
            verified against.
        chunk_size: Target rows per output chunk.
        num_chunks: Output chunks implied by *total_rows* and *chunk_size*.
        original_files: The split's source files, in sorted order.
        original_row_counts: Row count per source file, keyed by path.
    """

    split_type: str
    total_rows: int
    chunk_size: int
    num_chunks: int
    original_files: List[str]
    original_row_counts: Dict[str, int]


def get_parquet_files(split_dir: str, split_type: str) -> List[str]:
    """List a split's shards in a stable order, so ``file_id`` means the same thing every run.

    Args:
        split_dir: Directory holding the ``{split}_*.parquet`` shards.
        split_type: Split prefix to match, e.g. ``train``.

    Returns:
        The matching paths, sorted; empty when the split is absent.
    """
    pattern = os.path.join(split_dir, f"{split_type}_*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def count_rows_in_file(file_path: str) -> int:
    """Read a row count from parquet metadata, so index building costs no full scan.

    Args:
        file_path: Parquet file to measure.

    Returns:
        The file's row count.
    """
    lazy_df = pl.scan_parquet(file_path)
    count: int = lazy_df.select(pl.len()).collect().item()
    return count


def get_split_info(
    split_dir: str, split_type: str, target_chunk_size: int
) -> SplitInfo:
    """Plan the shuffle in one metadata pass, so nothing downstream has to re-scan.

    Args:
        split_dir: Directory holding the split's shards.
        split_type: Split prefix to inspect, e.g. ``train``.
        target_chunk_size: Rows wanted per output chunk.

    Returns:
        A :class:`SplitInfo` carrying the file list, per-file counts and the
        resulting chunk plan.

    Raises:
        ValueError: If the split has no files, since silently producing nothing
            would look like a successful run.
    """
    parquet_files = get_parquet_files(split_dir, split_type)

    if not parquet_files:
        raise ValueError(f"No {split_type} files found in {split_dir}")

    # Count rows in each file
    original_row_counts = {}
    total_rows = 0

    for file_path in parquet_files:
        row_count = count_rows_in_file(file_path)
        original_row_counts[file_path] = row_count
        total_rows += row_count

    # Calculate number of chunks needed
    num_chunks = (total_rows + target_chunk_size - 1) // target_chunk_size

    return SplitInfo(
        split_type=split_type,
        total_rows=total_rows,
        chunk_size=target_chunk_size,
        num_chunks=num_chunks,
        original_files=parquet_files,
        original_row_counts=original_row_counts,
    )


def create_row_indices(split_info: SplitInfo) -> List[RowIndex]:
    """Enumerate the split as addresses, the lightweight stand-in the shuffle operates on.

    Args:
        split_info: Plan produced by :func:`get_split_info`.

    Returns:
        One :class:`RowIndex` per row in the split, in file order.
    """
    indices = []

    for file_id, file_path in enumerate(split_info.original_files):
        row_count = split_info.original_row_counts[file_path]

        for row_index in range(row_count):
            indices.append(
                RowIndex(file_path=file_path, file_id=file_id, row_index=row_index)
            )

    return indices


def shuffle_indices(
    indices: List[RowIndex], seed: Optional[int] = None
) -> List[RowIndex]:
    """Randomise row order without moving any data.

    Args:
        indices: Row addresses to permute.
        seed: Fixes the permutation so a run can be reproduced.

    Returns:
        A shuffled copy; the input list is left untouched.
    """
    if seed is not None:
        random.seed(seed)

    shuffled = indices.copy()
    random.shuffle(shuffled)
    return shuffled


def chunk_indices(indices: List[RowIndex], chunk_size: int) -> List[List[RowIndex]]:
    """Decide the output shard boundaries before any data is read.

    Args:
        indices: Shuffled row addresses.
        chunk_size: Rows per output shard.

    Returns:
        One list of addresses per output shard.
    """
    chunks = []
    for i in range(0, len(indices), chunk_size):
        chunk = indices[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


def read_specific_rows(file_path: str, row_indices: List[int]) -> pl.DataFrame:
    """Fetch one chunk's share of a source file.

    Args:
        file_path: Source parquet file.
        row_indices: Zero-based offsets to keep.

    Returns:
        The selected rows, in the file's own order rather than the requested
        order — the shuffle comes from how chunks are composed, not from
        within-file ordering.
    """
    # Read the entire file and select specific rows
    df = pl.read_parquet(file_path)
    return df.filter(pl.arange(0, pl.len()).is_in(row_indices))


def group_indices_by_file(
    shuffled_chunks: List[List[RowIndex]],
) -> Dict[str, Dict[int, List[int]]]:
    """Invert the plan to file-major order, turning scattered row reads into one read per file and chunk.

    Args:
        shuffled_chunks: Output chunks expressed as row addresses.

    Returns:
        A ``{file_path: {chunk_id: [row_index, ...]}}`` lookup.
    """
    file_groups: Dict[str, Dict[int, List[int]]] = {}

    for chunk_id, chunk_indices in enumerate(shuffled_chunks):
        for row_index in chunk_indices:
            if row_index.file_path not in file_groups:
                file_groups[row_index.file_path] = {}
            if chunk_id not in file_groups[row_index.file_path]:
                file_groups[row_index.file_path][chunk_id] = []
            file_groups[row_index.file_path][chunk_id].append(row_index.row_index)

    return file_groups


def read_chunk_data(
    file_groups: Dict[str, Dict[int, List[int]]], chunk_id: int
) -> List[pl.DataFrame]:
    """Gather one output chunk's rows from every file that contributes to it.

    Args:
        file_groups: Lookup from :func:`group_indices_by_file`.
        chunk_id: Output chunk being assembled.

    Returns:
        One frame per contributing file, ready to concatenate.
    """
    chunk_data = []

    for file_path, chunk_data_dict in file_groups.items():
        if chunk_id in chunk_data_dict:
            row_indices = chunk_data_dict[chunk_id]
            if row_indices:  # Only read if there are rows from this file
                df = read_specific_rows(file_path, row_indices)
                chunk_data.append(df)

    return chunk_data


def write_chunk_file(
    chunk_data: List[pl.DataFrame], output_dir: str, split_type: str, chunk_id: int
) -> None:
    """Persist one assembled chunk under the naming scheme the rest of the pipeline expects.

    Args:
        chunk_data: Per-file frames making up the chunk; an empty list writes
            nothing rather than an empty file.
        output_dir: Destination directory.
        split_type: Split prefix used in the output filename.
        chunk_id: Index used in the output filename.
    """
    if not chunk_data:
        return

    # Combine all data for this chunk
    if len(chunk_data) == 1:
        final_chunk = chunk_data[0]
    else:
        final_chunk = pl.concat(chunk_data, how="vertical_relaxed")

    # Write chunk
    output_file = os.path.join(output_dir, f"{split_type}_{chunk_id}.parquet")
    final_chunk.write_parquet(output_file)

    logger.info(f"  Wrote {len(final_chunk):,} rows to {os.path.basename(output_file)}")


def create_shuffled_chunks(
    split_info: SplitInfo, shuffled_chunks: List[List[RowIndex]], output_dir: str
) -> None:
    """Materialise the shuffled plan on disk, one chunk at a time to bound memory.

    Args:
        split_info: Plan describing the split being shuffled.
        shuffled_chunks: Output chunks expressed as row addresses.
        output_dir: Destination for the shuffled shards; created if absent.
    """
    # Group indices by file for efficient reading
    file_groups = group_indices_by_file(shuffled_chunks)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each chunk
    for chunk_id in tqdm(range(len(shuffled_chunks)), desc="Creating shuffled chunks"):
        chunk_data = read_chunk_data(file_groups, chunk_id)
        write_chunk_file(chunk_data, output_dir, split_info.split_type, chunk_id)


def verify_row_counts(original_split_info: SplitInfo, output_dir: str) -> bool:
    """Catch silently dropped rows before the shuffled output is trusted for training.

    Args:
        original_split_info: Plan carrying the expected row total.
        output_dir: Directory holding the freshly-written shards.

    Returns:
        True when the totals agree; the caller reports the mismatch rather than
        raising, so remaining splits still get processed.
    """
    logger.info("Verifying row counts...")

    # Count rows in new chunks
    new_files = get_parquet_files(output_dir, original_split_info.split_type)
    new_total_rows = 0

    for file_path in new_files:
        row_count = count_rows_in_file(file_path)
        new_total_rows += row_count
        logger.info(f"  {os.path.basename(file_path)}: {row_count:,} rows")

    logger.info(f"Original total: {original_split_info.total_rows:,} rows")
    logger.info(f"New total: {new_total_rows:,} rows")

    if new_total_rows == original_split_info.total_rows:
        logger.info("Row counts match!")
        return True
    else:
        logger.error("Row counts do not match!")
        return False


def shuffle_split_by_indices(
    split_dir: str,
    split_type: str,
    chunk_size: int = 400000,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Shuffle one split without ever holding its data in memory.

    Runs the whole index pipeline — plan, enumerate, shuffle, chunk, write,
    verify — reporting progress at each step because the read-per-chunk pattern
    makes this slow on large splits.

    Args:
        split_dir: Directory containing the split files.
        split_type: Type of split (train, valid, test).
        chunk_size: Target number of rows per output chunk.
        seed: Random seed, for a reproducible permutation.
        output_dir: Where shuffled shards are written; defaults to *split_dir*,
            which overwrites the originals.
    """
    logger.info(f"Processing {split_type} split in {split_dir}")

    # Use output_dir if provided, otherwise use split_dir
    if output_dir is None:
        output_dir = split_dir

    # Step 1: Get split information
    logger.info("Step 1: Analysing split...")
    split_info = get_split_info(split_dir, split_type, chunk_size)

    logger.info(f"Found {len(split_info.original_files)} files:")
    for file_path in split_info.original_files:
        row_count = split_info.original_row_counts[file_path]
        logger.info(f"  {os.path.basename(file_path)}: {row_count:,} rows")

    logger.info(f"Total rows: {split_info.total_rows:,}")
    logger.info(f"Will create {split_info.num_chunks} chunks of ~{chunk_size:,} rows each")

    # Step 2: Create row indices
    logger.info("Step 2: Creating row indices...")
    indices = create_row_indices(split_info)
    logger.info(f"Created {len(indices):,} row indices")

    # Step 3: Shuffle indices
    logger.info("Step 3: Shuffling indices...")
    shuffled_indices = shuffle_indices(indices, seed)

    # Step 4: Chunk the shuffled indices
    logger.info("Step 4: Creating index chunks...")
    index_chunks = chunk_indices(shuffled_indices, chunk_size)
    logger.info(f"Created {len(index_chunks)} index chunks")

    # Step 5: Create shuffled chunks
    logger.info("Step 5: Creating shuffled chunks...")
    create_shuffled_chunks(split_info, index_chunks, output_dir)

    # Step 6: Verify row counts
    logger.info("Step 6: Verifying results...")
    success = verify_row_counts(split_info, output_dir)

    if success:
        logger.info(f"Successfully shuffled {split_type} split!")
    else:
        logger.error(f"Error in {split_type} split shuffling!")


def shuffle_all_splits(
    base_dir: str,
    chunk_size: int = 400000,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Shuffle every ``*_splits`` dataset folder under one root in a single run.

    Each split directory's layout is preserved in the output, and a failure on
    one split is reported and skipped so one bad dataset does not abort the
    whole sweep.

    Args:
        base_dir: Root containing per-dataset folders named ``*_splits``.
        chunk_size: Target number of rows per output chunk.
        seed: Random seed, for a reproducible permutation.
        output_dir: Root for the shuffled output; defaults to *base_dir*, which
            overwrites the originals in place.
    """
    base_path = Path(base_dir)

    # Find all split directories
    split_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_splits")
    ]

    if not split_dirs:
        logger.info(f"No split directories found in {base_dir}")
        return

    logger.info(f"Found split directories: {[d.name for d in split_dirs]}")

    # Process each split directory
    for split_dir in split_dirs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {split_dir.name}")
        logger.info(f"{'=' * 60}")

        # Determine output directory for this split
        if output_dir is not None:
            split_output_dir = os.path.join(output_dir, split_dir.name)
        else:
            split_output_dir = str(split_dir)

        # Process each split type
        for split_type in ["train", "valid", "test"]:
            try:
                shuffle_split_by_indices(
                    str(split_dir), split_type, chunk_size, seed, split_output_dir
                )
            except Exception as e:
                logger.error(f"Error processing {split_type} in {split_dir.name}: {e}")
                continue


app = typer.Typer(
    help="Index-based shuffle of parquet files across splits",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Root containing *_splits folders (repeatable)",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Root for shuffled output"),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target rows per output chunk"),
    ] = 400000,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = 42,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Shuffle every dataset's train/valid/test shards by permuting row addresses."""
    configure_script_logging(verbose=verbose)

    for base in input_dir:
        logger.info(f"Index-based shuffle of parquet files in {base}")
        logger.info(f"Target chunk size: {chunk_size:,} rows")
        if seed is not None:
            logger.info(f"Random seed: {seed}")
        logger.info(f"Output directory: {output_dir}")
        shuffle_all_splits(str(base), chunk_size, seed, str(output_dir))

    logger.info("Shuffling complete!")


if __name__ == "__main__":
    start_time = time.time()
    app()
    end_time = time.time()
    logger.info(
        f"Time taken for index-based shuffling: {(end_time - start_time) / 3600:.2f} hours"
    )
