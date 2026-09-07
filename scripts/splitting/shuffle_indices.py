"""Index-based shuffling of parquet files across train/validation/test splits.

This approach:
1. Counts rows in each chunk file
2. Assigns unique indices to each row (file_id, row_index)
3. Shuffles the index list
4. Creates new chunks by reading specific rows from original files
5. Verifies row counts match original
"""

import polars as pl
import glob
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
from dataclasses import dataclass
import time


@dataclass
class RowIndex:
    """Represents a specific row in a specific file."""

    file_path: str
    file_id: int
    row_index: int


@dataclass
class SplitInfo:
    """Information about a split (train/valid/test)."""

    split_type: str
    total_rows: int
    chunk_size: int
    num_chunks: int
    original_files: List[str]
    original_row_counts: Dict[str, int]


def get_parquet_files(split_dir: str, split_type: str) -> List[str]:
    """Get all parquet files for a specific split type (train/valid/test)."""
    pattern = os.path.join(split_dir, f"{split_type}_*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def count_rows_in_file(file_path: str) -> int:
    """Count rows in a parquet file efficiently."""
    lazy_df = pl.scan_parquet(file_path)
    count: int = lazy_df.select(pl.len()).collect().item()
    return count


def get_split_info(
    split_dir: str, split_type: str, target_chunk_size: int
) -> SplitInfo:
    """Get information about a split including row counts and chunking."""
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
    """Create a list of all row indices for the split."""
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
    """Shuffle the list of row indices."""
    if seed is not None:
        random.seed(seed)

    shuffled = indices.copy()
    random.shuffle(shuffled)
    return shuffled


def chunk_indices(indices: List[RowIndex], chunk_size: int) -> List[List[RowIndex]]:
    """Split the shuffled indices into chunks."""
    chunks = []
    for i in range(0, len(indices), chunk_size):
        chunk = indices[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


def read_specific_rows(file_path: str, row_indices: List[int]) -> pl.DataFrame:
    """Read specific rows from a parquet file."""
    # Read the entire file and select specific rows
    df = pl.read_parquet(file_path)
    return df.filter(pl.arange(0, pl.len()).is_in(row_indices))


def group_indices_by_file(
    shuffled_chunks: List[List[RowIndex]],
) -> Dict[str, Dict[int, List[int]]]:
    """Group row indices by file and chunk for efficient reading."""
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
    """Read data for a specific chunk from all relevant files."""
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
    """Write chunk data to a parquet file."""
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

    print(f"  Wrote {len(final_chunk):,} rows to {os.path.basename(output_file)}")


def create_shuffled_chunks(
    split_info: SplitInfo, shuffled_chunks: List[List[RowIndex]], output_dir: str
) -> None:
    """Create shuffled chunk files by reading specific rows from original files."""
    # Group indices by file for efficient reading
    file_groups = group_indices_by_file(shuffled_chunks)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each chunk
    for chunk_id in tqdm(range(len(shuffled_chunks)), desc="Creating shuffled chunks"):
        chunk_data = read_chunk_data(file_groups, chunk_id)
        write_chunk_file(chunk_data, output_dir, split_info.split_type, chunk_id)


def verify_row_counts(original_split_info: SplitInfo, output_dir: str) -> bool:
    """Verify that the new chunks have the same total row count as the original."""
    print("Verifying row counts...")

    # Count rows in new chunks
    new_files = get_parquet_files(output_dir, original_split_info.split_type)
    new_total_rows = 0

    for file_path in new_files:
        row_count = count_rows_in_file(file_path)
        new_total_rows += row_count
        print(f"  {os.path.basename(file_path)}: {row_count:,} rows")

    print(f"Original total: {original_split_info.total_rows:,} rows")
    print(f"New total: {new_total_rows:,} rows")

    if new_total_rows == original_split_info.total_rows:
        print("✅ Row counts match!")
        return True
    else:
        print("❌ Row counts do not match!")
        return False


def shuffle_split_by_indices(
    split_dir: str,
    split_type: str,
    chunk_size: int = 400000,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Shuffle a split using the index-based approach.

    Args:
        split_dir: Directory containing the split files
        split_type: Type of split (train, valid, test)
        chunk_size: Target number of rows per chunk
        seed: Random seed for reproducibility
        output_dir: Output directory (if None, uses split_dir)
    """
    print(f"Processing {split_type} split in {split_dir}")

    # Use output_dir if provided, otherwise use split_dir
    if output_dir is None:
        output_dir = split_dir

    # Step 1: Get split information
    print("Step 1: Analyzing split...")
    split_info = get_split_info(split_dir, split_type, chunk_size)

    print(f"Found {len(split_info.original_files)} files:")
    for file_path in split_info.original_files:
        row_count = split_info.original_row_counts[file_path]
        print(f"  {os.path.basename(file_path)}: {row_count:,} rows")

    print(f"Total rows: {split_info.total_rows:,}")
    print(f"Will create {split_info.num_chunks} chunks of ~{chunk_size:,} rows each")

    # Step 2: Create row indices
    print("Step 2: Creating row indices...")
    indices = create_row_indices(split_info)
    print(f"Created {len(indices):,} row indices")

    # Step 3: Shuffle indices
    print("Step 3: Shuffling indices...")
    shuffled_indices = shuffle_indices(indices, seed)

    # Step 4: Chunk the shuffled indices
    print("Step 4: Creating index chunks...")
    index_chunks = chunk_indices(shuffled_indices, chunk_size)
    print(f"Created {len(index_chunks)} index chunks")

    # Step 5: Create shuffled chunks
    print("Step 5: Creating shuffled chunks...")
    create_shuffled_chunks(split_info, index_chunks, output_dir)

    # Step 6: Verify row counts
    print("Step 6: Verifying results...")
    success = verify_row_counts(split_info, output_dir)

    if success:
        print(f"✅ Successfully shuffled {split_type} split!")
    else:
        print(f"❌ Error in {split_type} split shuffling!")


def shuffle_all_splits(
    base_dir: str,
    chunk_size: int = 400000,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Shuffle all splits (train, valid, test) for all datasets using index-based approach.

    Args:
        base_dir: Base directory containing the split folders
        chunk_size: Target number of rows per chunk
        seed: Random seed for reproducibility
        output_dir: Output directory (if None, uses base_dir)
    """
    base_path = Path(base_dir)

    # Find all split directories
    split_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_splits")
    ]

    if not split_dirs:
        print(f"No split directories found in {base_dir}")
        return

    print(f"Found split directories: {[d.name for d in split_dirs]}")

    # Process each split directory
    for split_dir in split_dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing {split_dir.name}")
        print(f"{'=' * 60}")

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
                print(f"Error processing {split_type} in {split_dir.name}: {e}")
                continue


def main() -> None:
    """Main function to parse arguments and shuffle splits."""
    parser = argparse.ArgumentParser(
        description="Index-based shuffle of parquet files across splits"
    )
    parser.add_argument(
        "--base-dir",
        default="<data-root>",
        help="Base directory containing split folders (default: <data-root>)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400000,
        help="Target number of rows per chunk (default: 400000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_output",
        help="Output directory for shuffled files (if None, overwrites original files)",
    )

    args = parser.parse_args()

    print(f"Index-based shuffle of parquet files in {args.base_dir}")
    print(f"Target chunk size: {args.chunk_size:,} rows")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if args.output_dir is not None:
        print(f"Output directory: {args.output_dir}")
    else:
        print("⚠️  WARNING: Will overwrite original files!")

    shuffle_all_splits(args.base_dir, args.chunk_size, args.seed, args.output_dir)

    print("\nShuffling complete!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(
        f"Time taken for index-based shuffling: {(end_time - start_time) / 3600:.2f} hours"
    )
