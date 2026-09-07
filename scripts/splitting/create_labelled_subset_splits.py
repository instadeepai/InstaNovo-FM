import pandas as pd
import polars as pl
import logging
from typing import Dict, Set, List
import glob
import os
from datetime import timedelta
import time
import typer

app = typer.Typer()

# Module-level constants to avoid B008 errors
INPUT_DIR_OPTION = typer.Option(
    "<data-root>/hcfm",
    "--input-dir",
    "-i",
    help="Input directory containing parquet files",
)
OUTPUT_DIR_OPTION = typer.Option(
    "<data-root>/hcfm_splits",
    "--output-dir",
    "-o",
    help="Output directory for split files",
)
SPLIT_ASSIGNMENTS_OPTION = typer.Option(
    "<data-root>/output_files/split_assignments.csv",
    "--split-assignments",
    "-s",
    help="Path to split assignments CSV file",
)
IDENTITY_BLACKLIST_OPTION = typer.Option(
    "splits/identity_splits_blacklist.csv",
    "--identity-blacklist",
    "-b",
    help="Path to identity blacklist CSV file",
)
CLASH_BLACKLIST_OPTION = typer.Option(
    "splits/clash_blacklist.csv",
    "--clash-blacklist",
    "-c",
    help="Path to clash blacklist CSV file",
)
ROWS_PER_FILE_OPTION = typer.Option(
    400_000,
    "--rows-per-file",
    "-r",
    help="Number of rows per output file",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose logging",
)
INPUT_DIRS_ARG = typer.Argument(
    ...,
    help="Input directories to process",
)
OUTPUT_BASE_OPTION = typer.Option(
    "<data-root>",
    "--output-base",
    "-o",
    help="Base directory for output",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = 42


def replace_i_with_l(peptide: str) -> str:
    """Replace all I's with L's in a peptide sequence."""
    return peptide.replace("I", "L")


def load_split_assignments(split_file: str) -> Dict[str, Set[str]]:
    """Load peptide assignments from the split assignments file."""
    if not os.path.exists(split_file):
        logger.error(f"Split assignments file {split_file} does not exist!")
        return {}

    df = pd.read_csv(split_file)
    splits: Dict[str, Set[str]] = {"train": set(), "test": set(), "valid": set()}

    for _, row in df.iterrows():
        peptide = replace_i_with_l(row["sequence"])
        split = row["split"]
        splits[split].add(peptide)

    logger.info("Loaded split assignments:")
    for split, peptides in splits.items():
        logger.info(f"  {split}: {len(peptides)} peptides")

    return splits


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    return str(timedelta(seconds=int(seconds)))


def write_buffer(
    split: str,
    split_buffers: Dict[str, list],
    file_counters: Dict[str, int],
    buffer_sizes: Dict[str, int],
    output_dir: str,
) -> None:
    """Write buffer to file and clear it."""
    if split_buffers[split]:
        # Concatenate all data in buffer
        combined_data = pl.concat(split_buffers[split], how="vertical_relaxed")
        # Shuffle the data
        combined_data = combined_data.sample(fraction=1.0, seed=SEED, shuffle=True)
        output_path = os.path.join(
            output_dir, f"{split}_{file_counters[split]}.parquet"
        )
        combined_data.write_parquet(output_path)
        logger.info(f"Wrote {len(combined_data):,} rows to {output_path}")
        split_buffers[split] = []
        buffer_sizes[split] = 0
        file_counters[split] += 1


def load_blacklists(
    identity_blacklist_file: str, clash_blacklist_file: str
) -> tuple[Set[str], Set[str]]:
    """Load both blacklist files and return sets of normalised peptides."""
    identity_blacklist = set()
    clash_blacklist = set()

    # Load identity blacklist (for training exclusion)
    if os.path.exists(identity_blacklist_file):
        df = pd.read_csv(identity_blacklist_file)
        identity_blacklist = {replace_i_with_l(seq) for seq in df["sequence"]}
        logger.info(
            f"Loaded {len(identity_blacklist)} peptides from identity blacklist"
        )
    else:
        logger.warning(f"Identity blacklist file {identity_blacklist_file} not found")

    # Load clash blacklist (for complete exclusion)
    if os.path.exists(clash_blacklist_file):
        df = pd.read_csv(clash_blacklist_file)
        clash_blacklist = {replace_i_with_l(seq) for seq in df["sequence"]}
        logger.info(f"Loaded {len(clash_blacklist)} peptides from clash blacklist")
    else:
        logger.warning(f"Clash blacklist file {clash_blacklist_file} not found")

    return identity_blacklist, clash_blacklist


def normalise_dataframe_schema(
    df: pl.DataFrame, reference_schema: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """Add missing columns with appropriate null values based on reference schema and ensure correct column order."""
    # Add missing columns with null values
    for col_name, dtype in reference_schema.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

    # Reorder columns to match reference schema
    return df.select(list(reference_schema.keys()))


def filter_spectra(df: pl.LazyFrame) -> pl.LazyFrame:
    """Apply filtering criteria to spectra."""
    return df.filter(
        (pl.col("retention_time") <= 10800)  # Filter retention times above 10800
        & (pl.col("lower_offset") <= 300)  # Filter lower offsets above 300
        & (pl.col("precursor_charge") > 0)  # Filter precursor charges of 0
        & (pl.col("precursor_charge") <= 7)  # Filter precursor charges above 7
        & (pl.col("precursor_mz") <= 2000)  # Filter precursor mz values above 2000
    )


def process_single_file(
    file_path: str,
    split_lookup: Dict[str, Set[str]],
    split_buffers: Dict[str, list],
    buffer_sizes: Dict[str, int],
    rows_per_file: int,
    output_dir: str,
    file_counters: Dict[str, int],
    identity_blacklist: Set[str],
    clash_blacklist: Set[str],
) -> int:
    """Process a single file and distribute its data to appropriate split buffers."""
    reference_schema = {
        "index": pl.Int64,
        "scan": pl.String,
        "header": pl.String,
        "retention_time": pl.Float64,
        "frag_type": pl.String,
        "collision_energy": pl.Float64,
        "precursor_mz": pl.Float64,
        "precursor_charge": pl.Int64,
        "precursor_intensity": pl.Float64,
        "lower_offset": pl.Float64,
        "upper_offset": pl.Float64,
        "isolation_target_old": pl.Float64,
        "mz_array": pl.List(pl.Float64),
        "intensity_array": pl.List(pl.Float32),
        "scale_factor": pl.Float32,
        "modified_peptide": pl.String,
        "peptide_observed_mz": pl.Float64,
        "peptide_calc_mz": pl.Float64,
        "delta_mass": pl.Float64,
        "retention": pl.Float64,
        "expectation": pl.Float64,
        "hyperscore": pl.Float64,
        "nextscore": pl.Float64,
        "probability": pl.Float64,
        "auc_intensity": pl.Float64,
        "protein": pl.String,
        "experiment_name": pl.String,
        "isolation_target": pl.Float64,
        "unmodified_peptide": pl.String,
        "sequence": pl.String,
        "filepath": pl.String,
        "normalised_peptide": pl.String,
    }

    # Read file and add normalised peptide column
    df = pl.scan_parquet(file_path)

    # Apply filtering criteria
    df = filter_spectra(df)

    df = df.with_columns(
        [
            pl.col("peptide")
            .map_elements(replace_i_with_l, return_dtype=pl.String)
            .alias("normalised_peptide"),
            pl.lit(file_path).alias("filepath"),
        ]
    )

    # Count spectra from clash blacklisted peptides
    excluded_spectra: int = (
        df.filter(pl.col("normalised_peptide").is_in(clash_blacklist))
        .select(pl.len())
        .collect()
        .item()
    )

    # Process each split
    for split in ["train", "test", "valid"]:
        split_peptides = split_lookup[split]

        # Filter for current split and collect
        split_data = df.filter(
            pl.col("normalised_peptide").is_in(split_peptides)
        ).collect()

        if not split_data.is_empty():
            # Normalise schema
            split_data = normalise_dataframe_schema(split_data, reference_schema)

            # Apply blacklist rules
            if split == "train":
                # For training, exclude both identity and clash blacklisted peptides
                split_data = split_data.filter(
                    ~pl.col("normalised_peptide").is_in(identity_blacklist)
                    & ~pl.col("normalised_peptide").is_in(clash_blacklist)
                )
            else:
                # For test/valid, only exclude clash blacklisted peptides
                split_data = split_data.filter(
                    ~pl.col("normalised_peptide").is_in(clash_blacklist)
                )

            if not split_data.is_empty():
                # Add to buffer
                split_buffers[split].append(split_data)
                buffer_sizes[split] += len(split_data)

                # If buffer exceeds threshold, write to file
                if buffer_sizes[split] >= rows_per_file:
                    write_buffer(
                        split,
                        split_buffers,
                        file_counters,
                        buffer_sizes,
                        output_dir,
                    )

    return excluded_spectra


def process_and_write_files(
    parquet_files: list[str],
    split_lookup: Dict[str, Set[str]],
    output_dir: str,
    rows_per_file: int,
    identity_blacklist: Set[str],
    clash_blacklist: Set[str],
) -> int:
    """Process files and write split data."""
    # Initialise buffers and file counters for each split
    split_buffers: Dict[str, list] = {
        "train": [],
        "test": [],
        "valid": [],
    }
    file_counters: Dict[str, int] = {"train": 0, "test": 0, "valid": 0}
    buffer_sizes: Dict[str, int] = {"train": 0, "test": 0, "valid": 0}

    # Process files and write splits
    logger.info("Processing files and writing splits")
    total_files = len(parquet_files)
    start_time = time.time()
    total_excluded_spectra = 0

    for i, file_path in enumerate(parquet_files):
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            files_per_second = (i + 1) / elapsed_time
            remaining_files = total_files - (i + 1)
            estimated_remaining_time = (
                remaining_files / files_per_second if files_per_second > 0 else 0
            )

            logger.info(
                f"Processing file {i + 1} / {total_files} ({(i + 1) / total_files * 100:.1f}%) - "
                f"Elapsed: {format_time(elapsed_time)} - "
                f"Est. remaining: {format_time(estimated_remaining_time)}"
            )

        excluded_spectra = process_single_file(
            file_path,
            split_lookup,
            split_buffers,
            buffer_sizes,
            rows_per_file,
            output_dir,
            file_counters,
            identity_blacklist,
            clash_blacklist,
        )
        total_excluded_spectra += excluded_spectra

    # Write remaining buffers
    logger.info("Writing remaining buffers")
    for split in ["train", "test", "valid"]:
        if split_buffers[split]:
            write_buffer(
                split,
                split_buffers,
                file_counters,
                buffer_sizes,
                output_dir,
            )

    return total_excluded_spectra


def count_spectra(
    input_dir: str,
    output_dir: str,
    excluded_spectra: int,
) -> None:
    """Count spectra in each split and verify against filtered input."""
    logger.info("Final spectra distribution:")
    split_spectra_counts = {}
    for split in ["train", "test", "valid"]:
        split_files = glob.glob(os.path.join(output_dir, f"{split}_*.parquet"))
        split_count = 0
        for file in split_files:
            # Select index column to count spectra without loading all data into memory
            df = pl.scan_parquet(file).select(["index"]).collect()
            split_count += len(df)
        split_spectra_counts[split] = split_count

    total_spectra = sum(split_spectra_counts.values())
    for split, count in split_spectra_counts.items():
        percentage = (count / (total_spectra + excluded_spectra)) * 100
        logger.info(f"\t{split}: {count:,} spectra ({percentage:.1f}%)")

    # Verify spectra counts
    logger.info("Verifying total spectra counts")
    total_input_spectra = 0
    total_filtered_spectra = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                # Count total spectra
                df = pl.scan_parquet(file_path)
                total_input_spectra += df.select(pl.len()).collect().item()

                # Count filtered spectra
                filtered_df = filter_spectra(df)
                total_filtered_spectra += filtered_df.select(pl.len()).collect().item()

    total_output_spectra = sum(split_spectra_counts.values())

    logger.info(f"Total input spectra: {total_input_spectra:,}")
    logger.info(f"Total filtered spectra: {total_filtered_spectra:,}")
    logger.info(f"Total output spectra: {total_output_spectra:,}")
    logger.info(f"Excluded spectra (from clash blacklist): {excluded_spectra:,}")
    logger.info(
        f"Total (output + excluded): {total_output_spectra + excluded_spectra:,}"
    )

    # Verify that output + excluded matches filtered input
    if total_filtered_spectra == total_output_spectra + excluded_spectra:
        logger.info(
            "Verification passed: Filtered input spectra count matches output + excluded spectra"
        )
    else:
        logger.error(
            f"Verification failed: {total_filtered_spectra:,} filtered input spectra vs "
            f"{total_output_spectra:,} in output + {excluded_spectra:,} excluded"
        )
        raise ValueError("Spectra count verification failed!")


def create_subset_splits(
    input_dir: str,
    output_dir: str,
    split_assignments_file: str,
    identity_blacklist_file: str,
    clash_blacklist_file: str,
    rows_per_file: int = 400_000,
) -> None:
    """Create splits for a subset dataset based on the split assignments from lcfm."""
    # Load split assignments
    logger.info("Loading split assignments...")
    split_lookup = load_split_assignments(split_assignments_file)
    if not split_lookup:
        logger.error("No split assignments loaded, cannot proceed!")
        return

    # Load blacklists
    logger.info("Loading blacklists...")
    identity_blacklist, clash_blacklist = load_blacklists(
        identity_blacklist_file, clash_blacklist_file
    )

    # Find all parquet files
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    logger.info(f"Found {len(parquet_files)} parquet files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process files and write splits
    excluded_spectra = process_and_write_files(
        parquet_files,
        split_lookup,
        output_dir,
        rows_per_file,
        identity_blacklist,
        clash_blacklist,
    )

    # Verify and count spectra
    count_spectra(input_dir, output_dir, excluded_spectra)


@app.command()
def subset(
    input_dir: str = INPUT_DIR_OPTION,
    output_dir: str = OUTPUT_DIR_OPTION,
    split_assignments_file: str = SPLIT_ASSIGNMENTS_OPTION,
    identity_blacklist_file: str = IDENTITY_BLACKLIST_OPTION,
    clash_blacklist_file: str = CLASH_BLACKLIST_OPTION,
    rows_per_file: int = ROWS_PER_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Create subset splits based on existing split assignments."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo(f"Processing subset files from {input_dir}")
    typer.echo(f"Output will be written to {output_dir}")
    typer.echo(f"Using split assignments from {split_assignments_file}")

    create_subset_splits(
        input_dir,
        output_dir,
        split_assignments_file,
        identity_blacklist_file,
        clash_blacklist_file,
        rows_per_file,
    )


@app.command()
def batch(
    input_dirs: List[str] = INPUT_DIRS_ARG,
    output_base: str = OUTPUT_BASE_OPTION,
    split_assignments_file: str = SPLIT_ASSIGNMENTS_OPTION,
    identity_blacklist_file: str = IDENTITY_BLACKLIST_OPTION,
    clash_blacklist_file: str = CLASH_BLACKLIST_OPTION,
    rows_per_file: int = ROWS_PER_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Process multiple input directories in batch for subset splits."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            typer.echo(f"Warning: Input directory {input_dir} does not exist, skipping")
            continue

        # Create output directory name based on input directory name
        input_name = os.path.basename(input_dir.rstrip("/"))
        output_dir = os.path.join(output_base, f"{input_name}_splits")

        typer.echo(f"Processing {input_dir} -> {output_dir}")

        create_subset_splits(
            input_dir,
            output_dir,
            split_assignments_file,
            identity_blacklist_file,
            clash_blacklist_file,
            rows_per_file,
        )



if __name__ == "__main__":
    app()
