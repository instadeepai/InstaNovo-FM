import pandas as pd
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalise_sequence(sequence: str) -> str:
    """Collapse I/L ambiguity by converting L to I.

    Mind the direction. This maps to ``I``; the peptide registry and the splitting
    pipeline map to ``L`` (``split_labelled_data.py`` builds ``normalised_peptide``
    with ``str.replace_all("I", "L")``, and the published registry's ``peptide``
    column contains no ``I`` at all). Both directions define the same equivalence
    classes -- a peptide and its I/L variants land together either way -- so
    grouping with either is correct, and this script is self-consistent because it
    only ever compares keys it produced itself from its own CSV inputs.

    The two conventions are not interchangeable as *keys*, though. A key from here
    holds ``I`` where a registry key holds ``L``, so looking one up against the
    registry matches nothing rather than matching something subtly wrong -- a
    silent empty result, which is easy to misread as "this peptide is new".

    So if this output is ever joined against the registry, convert to the registry's
    direction at the join. Do not flip this function: that would change the keys
    written into this script's own outputs and the files already derived from them.
    """
    return sequence.replace("L", "I")


def load_split_files(csv_files: List[str]) -> Dict[str, pd.DataFrame]:
    """Load all split CSV files into a dictionary."""
    split_dfs = {}
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
            split_dfs[file] = df
            logger.info(f"Loaded {len(df)} peptides from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    return split_dfs


def find_overlapping_peptides(
    split_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Find peptides that appear in multiple splits across files."""
    # Dictionary to store peptide -> (file, split) mappings
    peptide_assignments: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    # Collect all peptide assignments
    for file, df in split_dfs.items():
        for _, row in df.iterrows():
            peptide = row["normalised_sequence"]  # Use normalised sequence
            original_peptide = row["sequence"]  # Keep original for display
            split = row["split"]
            peptide_assignments[peptide].append((file, split, original_peptide))

    # Find peptides with multiple assignments
    overlapping_peptides = {
        peptide: assignments
        for peptide, assignments in peptide_assignments.items()
        if len(assignments) > 1
    }

    return overlapping_peptides


def get_all_files(
    overlapping_peptides: Dict[str, List[Tuple[str, str, str]]],
) -> List[str]:
    """Get sorted list of all files that contain overlapping peptides."""
    return sorted(
        {file for assignments in overlapping_peptides.values() for file, _, _ in assignments}
    )


def group_assignments(
    overlapping_peptides: Dict[str, List[Tuple[str, str, str]]],
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, str]]]], Dict[str, List[Tuple[str, str, str]]]]:
    """Group peptide assignments by file and create a flat list of all assignments."""
    file_assignments: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    peptide_assignments: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    for peptide, assignments in overlapping_peptides.items():
        for file, split, original_peptide in assignments:
            file_assignments[file][peptide].append((split, original_peptide))
            peptide_assignments[peptide].append((file, split, original_peptide))

    return file_assignments, peptide_assignments


def report_within_file_clashes(
    file_assignments: Dict[str, Dict[str, List[Tuple[str, str]]]],
) -> bool:
    """Report any clashing assignments within the same file."""
    has_clashes = False
    logger.info("\n=== Within-file clashes ===")

    for file, peptide_splits in file_assignments.items():
        clashes = {
            peptide: splits
            for peptide, splits in peptide_splits.items()
            if len({split for split, _ in splits}) > 1
        }

        if clashes:
            has_clashes = True
            logger.info(f"\nClashing assignments in {file}:")
            for peptide, splits in clashes.items():
                split_list = [f"{split} ({orig})" for split, orig in splits]
                logger.info(f"  {peptide}: {', '.join(split_list)}")

    if not has_clashes:
        logger.info("No clashing assignments found within files!")

    return has_clashes


def create_conflict_row(
    peptide: str, assignments: List[Tuple[str, str, str]], all_files: List[str]
) -> Dict[str, str]:
    """Create a row for the conflicts CSV file."""
    row = {"normalised_sequence": peptide}

    for file in all_files:
        file_assignments_list = [(split, orig) for f, split, orig in assignments if f == file]
        if file_assignments_list:
            splits_str = "; ".join(f"{split} ({orig})" for split, orig in file_assignments_list)
            logger.info(f"  - {splits_str} in {file}")
            row[file] = splits_str
        else:
            row[file] = "not present"

    return row


def report_cross_file_conflicts(
    peptide_assignments: Dict[str, List[Tuple[str, str, str]]], all_files: List[str]
) -> List[Dict[str, str]]:
    """Report conflicts between files and prepare CSV data."""
    logger.info("\n=== Cross-file split conflicts ===")
    has_cross_conflicts = False
    csv_rows = []

    for peptide, assignments in peptide_assignments.items():
        unique_splits = {split for _, split, _ in assignments}
        if len(unique_splits) > 1:
            has_cross_conflicts = True
            logger.info(f"\nPeptide {peptide} has conflicting splits across files:")
            row = create_conflict_row(peptide, assignments, all_files)
            csv_rows.append(row)

    if not has_cross_conflicts:
        logger.info("No cross-file split conflicts found!")

    return csv_rows


def save_conflicts_to_csv(csv_rows: List[Dict[str, str]]) -> None:
    """Save conflict information to a CSV file."""
    if csv_rows:
        output_file = "output_files/split_conflicts.csv"
        df = pd.DataFrame(csv_rows)
        df.to_csv(output_file, index=False)
        logger.info(f"\nDetailed results written to {output_file}")


def analyse_overlaps(
    overlapping_peptides: Dict[str, List[Tuple[str, str, str]]],
) -> None:
    """Analyse and report overlapping peptides, focusing on both within-file clashes and cross-file conflicts."""
    if not overlapping_peptides:
        logger.info("No overlapping peptides found!")
        return

    all_files = get_all_files(overlapping_peptides)
    file_assignments, peptide_assignments = group_assignments(overlapping_peptides)

    report_within_file_clashes(file_assignments)
    csv_rows = report_cross_file_conflicts(peptide_assignments, all_files)
    save_conflicts_to_csv(csv_rows)


def main() -> None:
    """Main function to verify split assignments."""
    # List of split files to check
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

    # Find overlapping peptides
    logger.info("\nChecking for overlapping peptides...")
    overlapping_peptides = find_overlapping_peptides(split_dfs)

    # Analyse and report overlaps
    analyse_overlaps(overlapping_peptides)


if __name__ == "__main__":
    main()
