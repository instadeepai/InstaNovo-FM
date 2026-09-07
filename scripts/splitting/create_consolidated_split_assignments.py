import pandas as pd
import logging
from typing import Dict, List, Set, TypedDict, Tuple
from collections import defaultdict
import typer

app = typer.Typer()

# Module-level constants to avoid B008 errors
SPLIT_FILES_ARG = typer.Argument(
    ...,
    help="List of split CSV files to consolidate",
)
OUTPUT_FILE_OPTION = typer.Option(
    "splits/consolidated_splits.csv",
    "--output-file",
    "-o",
    help="Output file for consolidated splits",
)
BLACKLIST_FILE_OPTION = typer.Option(
    "splits/clash_blacklist.csv",
    "--blacklist-file",
    "-b",
    help="Output file for clash blacklist",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose logging",
)
INPUT_PATTERN_ARG = typer.Argument(
    "splits/*_splits.csv",
    help="Glob pattern for input split files",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalise_sequence(sequence: str) -> str:
    """Normalise peptide sequence by converting L to I to handle I/L ambiguity."""
    return sequence.replace("L", "I")


class PeptideData(TypedDict):
    """Type definition for peptide assignment data."""

    original_sequences: Set[str]
    assignments: Set[str]
    file_assignments: Dict[str, Set[str]]


def check_duplicate_assignments(df: pd.DataFrame, file: str) -> None:
    """Check for and report duplicate peptides with conflicting assignments."""
    norm_duplicates = df[df.duplicated(["normalised_sequence"], keep=False)]
    if not norm_duplicates.empty:
        for norm_seq, group in norm_duplicates.groupby("normalised_sequence"):
            unique_splits = set(group["split"])
            if len(unique_splits) > 1:
                orig_seqs = group["sequence"].tolist()
                splits = group["split"].tolist()
                logger.warning(
                    f"Found conflicting assignments in {file} for {norm_seq}:"
                )
                logger.warning(f"  {list(zip(orig_seqs, splits))}")


def load_split_files(csv_files: List[str]) -> Dict[str, pd.DataFrame]:
    """Load all split CSV files into a dictionary."""
    split_dfs: Dict[str, pd.DataFrame] = {}
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "sequence" not in df.columns:
                logger.error(f"File {file} does not contain a 'sequence' column")
                continue
            if "split" not in df.columns:
                logger.error(f"File {file} does not contain a 'split' column")
                continue

            # Normalise sequences
            df["normalised_sequence"] = df["sequence"].apply(normalise_sequence)

            # Check for duplicates with conflicting assignments
            check_duplicate_assignments(df, file)

            # Remove duplicates, keeping the first occurrence
            df = df.drop_duplicates(["normalised_sequence"], keep="first")
            split_dfs[file] = df
            logger.info(f"Loaded {len(df)} unique peptides from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    return split_dfs


def collect_peptide_assignments(
    split_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, PeptideData]:
    """Collect all peptide assignments from the split files."""
    peptide_assignments: Dict[str, PeptideData] = defaultdict(
        lambda: {
            "original_sequences": set(),
            "assignments": set(),
            "file_assignments": defaultdict(set),
        }
    )

    for file, df in split_dfs.items():
        for _, row in df.iterrows():
            peptide = row["normalised_sequence"]
            original = row["sequence"]
            split = row["split"]
            peptide_assignments[peptide]["original_sequences"].add(original)
            peptide_assignments[peptide]["assignments"].add(split)
            peptide_assignments[peptide]["file_assignments"][file].add(split)

    return peptide_assignments


def identify_clashing_peptides(peptide_assignments: Dict[str, PeptideData]) -> Set[str]:
    """Identify peptides that have conflicting assignments across files."""
    clashing_peptides = set()
    for peptide, data in peptide_assignments.items():
        # Check for conflicts within the same file
        for _file, file_splits in data["file_assignments"].items():
            if len(file_splits) > 1:
                clashing_peptides.add(peptide)
                break

        # Check for conflicts across files
        if len(data["assignments"]) > 1:
            clashing_peptides.add(peptide)

    return clashing_peptides


def create_consolidated_splits(
    split_dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create consolidated splits following the specified rules."""
    peptide_assignments = collect_peptide_assignments(split_dfs)

    # Identify peptides with clashes
    clashing_peptides = identify_clashing_peptides(peptide_assignments)
    logger.info(f"Found {len(clashing_peptides)} peptides with assignment clashes")

    # Create blacklist DataFrame
    blacklist_rows = []
    for peptide in clashing_peptides:
        data = peptide_assignments[peptide]
        for original_seq in data["original_sequences"]:
            blacklist_rows.append(
                {
                    "sequence": original_seq,
                    "normalised_sequence": peptide,
                    "conflicting_assignments": ", ".join(sorted(data["assignments"])),
                }
            )
    blacklist_df = pd.DataFrame(blacklist_rows)

    # Create consolidated assignments (excluding clashing peptides)
    consolidated_rows: List[Dict[str, str]] = []
    for peptide, data in peptide_assignments.items():
        if peptide in clashing_peptides:
            continue

        assignments = data["assignments"]
        original_sequences = data["original_sequences"]

        # Use the first original sequence (they're equivalent after normalisation)
        original_sequence = next(iter(original_sequences))
        consolidated_rows.append(
            {"sequence": original_sequence, "split": next(iter(assignments))}
        )

    # Create DataFrame and sort by sequence
    consolidated_df = pd.DataFrame(consolidated_rows)
    consolidated_df = consolidated_df.sort_values("sequence")

    return consolidated_df, blacklist_df


def process_consolidated_splits(
    split_files: List[str],
    output_file: str = "splits/consolidated_splits.csv",
    blacklist_file: str = "splits/clash_blacklist.csv",
    verbose: bool = False,
) -> None:
    """Process split files and create consolidated splits."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo(f"Processing {len(split_files)} split files")
    typer.echo(f"Output will be written to {output_file}")
    typer.echo(f"Blacklist will be written to {blacklist_file}")

    # Load split files
    logger.info("Loading split files...")
    split_dfs = load_split_files(split_files)

    if not split_dfs:
        logger.error("No valid split files found!")
        return

    # Create consolidated splits
    logger.info("Creating consolidated splits...")
    consolidated_df, blacklist_df = create_consolidated_splits(split_dfs)

    # Save consolidated splits
    consolidated_df.to_csv(output_file, index=False)
    logger.info(f"Consolidated splits saved to {output_file}")

    # Save blacklist
    blacklist_df.to_csv(blacklist_file, index=False)
    logger.info(f"Blacklist saved to {blacklist_file}")

    # Print summary
    split_counts = consolidated_df["split"].value_counts()
    logger.info("Final split distribution:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count} peptides")

    logger.info(f"\nTotal peptides excluded: {len(blacklist_df)}")


@app.command()
def consolidate(
    split_files: List[str] = SPLIT_FILES_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    blacklist_file: str = BLACKLIST_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Consolidate multiple split files into a single consolidated split file."""
    process_consolidated_splits(
        split_files,
        output_file,
        blacklist_file,
        verbose,
    )


@app.command()
def default(
    output_file: str = OUTPUT_FILE_OPTION,
    blacklist_file: str = BLACKLIST_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Create consolidated splits using default split files."""
    # Default split files
    default_split_files = [
        "splits/identity_splits_phospho.csv",
        "splits/identity_splits_proteome_tools.csv",
        "splits/massivekb_splits.csv",
    ]

    typer.echo("Using default split files:")
    for file in default_split_files:
        typer.echo(f"  - {file}")

    process_consolidated_splits(
        default_split_files,
        output_file,
        blacklist_file,
        verbose,
    )


@app.command()
def batch(
    input_pattern: str = INPUT_PATTERN_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    blacklist_file: str = BLACKLIST_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Process all split files matching a pattern."""
    import glob

    split_files = glob.glob(input_pattern)
    if not split_files:
        typer.echo(f"No files found matching pattern: {input_pattern}")
        return

    typer.echo(f"Found {len(split_files)} files matching pattern: {input_pattern}")
    for file in split_files:
        typer.echo(f"  - {file}")

    process_consolidated_splits(
        split_files,
        output_file,
        blacklist_file,
        verbose,
    )


def main() -> None:
    """Main function to create consolidated splits."""
    # List of split files to process
    split_files = [
        "splits/identity_splits_phospho.csv",
        "splits/identity_splits_proteome_tools.csv",
        "splits/massivekb_splits.csv",
    ]

    # Load split files
    logger.info("Loading split files...")
    split_dfs = load_split_files(split_files)

    if not split_dfs:
        logger.error("No valid split files found!")
        return

    # Create consolidated splits
    logger.info("Creating consolidated splits...")
    consolidated_df, blacklist_df = create_consolidated_splits(split_dfs)

    # Save consolidated splits
    output_file = "splits/consolidated_splits.csv"
    consolidated_df.to_csv(output_file, index=False)
    logger.info(f"Consolidated splits saved to {output_file}")

    # Save blacklist
    blacklist_file = "splits/clash_blacklist.csv"
    blacklist_df.to_csv(blacklist_file, index=False)
    logger.info(f"Blacklist saved to {blacklist_file}")

    # Print summary
    split_counts = consolidated_df["split"].value_counts()
    logger.info("Final split distribution:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count} peptides")

    logger.info(f"\nTotal peptides excluded: {len(blacklist_df)}")


if __name__ == "__main__":
    app()
