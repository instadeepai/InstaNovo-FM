"""Translate EncyclopeDIA peptide modifications into ProForma UNIMOD labels.

Run this after modification inventories pass ``check_modifications.py`` so
training sequences use canonical labels. It is specific to LCFM projects and
combines gold-standard mappings with filename-dependent PXD009449 overrides.

The gold standard modifications file is a Excel file that contains the gold standard modifications for the LCFM project.
This means that the EncyclopeDIA modifications encountered in the LCFM project have a one-to-one or many-to-one mapping to UNIMOD modifications.

Some EncyclopeDIA modifications are ambiguous and can map to multiple UNIMOD modifications.
This is always a risk when translating modification labels from one scheme to another, and we have found this to be the case for some modifications in (at least)
the PXD009449 project. The PXD009449 ambiguous modifications file is a Excel file that contains the ambiguous modifications for the PXD009449 project where
we require file name information to be able to assign a UNIMOD modification.

Note that the gold standard modifications file may additionally contain some EncyclopeDIA modifications that we flagged as ambiguous in PXD009449.
This is because, based on the context of the other projects where these modifications were detected, we were able to assign a single UNIMOD label.
For example, the modification "K[242]" was detected in both PXD009449 and PXD037009, but we were able to assign it to "[UNIMOD:121]" (ubiquitination) for the project PXD037009.

Our intended workflow is to use the gold standard modifications file as the base for all projects.
For files in PXD009449, we check if the file name contains any modification_in_file_name patterns from the PXD009449 ambiguous modifications file.
If a match is found, we create file-specific modification dictionaries from the matching rows and merge them with the gold standard dictionaries,
with file-specific modifications taking precedence. This allows the same modification (e.g., "K[242]") to map to different UNIMOD encodings
based on the file name pattern (e.g., "[UNIMOD:1848]" for files containing "ubiquitin" vs "[UNIMOD:21]" for files containing "acetylation").
If no file name match is found for a PXD009449 file, we fall back to using only the gold standard modifications.

Note that all N- and C-terminal modifications are encoded with a trailing hyphen and a leading hyphen, respectively.
For example, the sequence "n[143]PEPTIDE" is encoded as "[UNIMOD:1]-PEPTIDE".

We expect the gold standard xlsx file to contain the columns:
- modification: The EncyclopeDIA modification, additionally containing the associated amino acid residue and optionally an "n" or "c" to indicate an N-terminal or C-terminal modification
- project_name: The project name
- file_name: The file name
- proposed_unimod_encoding: The proposed UNIMOD encoding, without any associated amino acid residue or "n" or "c"

We expect the PXD009449 ambiguous modifications xlsx file to contain the columns:
- modification: The EncyclopeDIA modification, additionally containing the associated amino acid residue and optionally an "n" or "c" to indicate an N-terminal or C-terminal modification
- project_name: The project name
- modification_in_file_name: The modification contained within the file name
- proposed_unimod_encoding: The proposed UNIMOD encoding, without any associated amino acid residue or "n" or "c"

We expect the parquet files to contain the columns:
- peptide: The peptide sequence
- modified_peptide: The modified peptide sequence (can be None)

CLI::

    uv run python -m scripts.preprocessing.label_modifications --help
    uv run python -m scripts.preprocessing.label_modifications --input-dir <subfolder>
    uv run python -m scripts.preprocessing.label_modifications --input-dir <subfolder-a> --input-dir <subfolder-b> \
        --gold-standard-mods assets/mod_dicts/gold_standard_modifications.xlsx \
        --ambiguous-mods assets/mod_dicts/PXD009449_ambiguous_mods.xlsx

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Annotated, Callable, List

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging
from scripts.paths import DEFAULT_AMBIGUOUS_MODS, DEFAULT_GOLD_STANDARD_MODS

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Label modifications in parquet files",
    no_args_is_help=True,
    add_completion=False,
)


def read_gold_standard_modifications(file_path: str) -> pl.DataFrame:
    """Load authoritative mappings before translating peptide annotations.

    Args:
        file_path: Gold-standard Excel workbook.

    Returns:
        Mapping rows used as defaults for all projects.
    """
    return pl.read_excel(file_path)


def read_pxd009449_ambiguous_modifications(file_path: str) -> pl.DataFrame:
    """Load filename-dependent overrides needed for ambiguous PXD009449 labels.

    Args:
        file_path: PXD009449 override Excel workbook.

    Returns:
        Mapping rows used only when filenames match their patterns.
    """
    return pl.read_excel(file_path)


def create_mod_dict(modification_df: pl.DataFrame) -> dict[str, str]:
    """Attach residues to UNIMOD labels so residue modifications can be replaced safely.

    Args:
        modification_df: Rows containing ``modification`` and
            ``proposed_unimod_encoding`` columns.

    Returns:
        EncyclopeDIA residue labels mapped to residue-qualified UNIMOD labels.

    Raises:
        ValueError: If one residue contains multiple unsupported modifications.
    """
    if modification_df.height == 0:
        return {}

    residue_mods = modification_df.filter(
        ~pl.col("modification").str.contains("n")
        & ~pl.col("modification").str.contains("c")
    )

    if len(residue_mods) == 0:
        return {}

    # Check for modifications with multiple modification brackets on a single amino acid
    # e.g., "K[123][456]" - these require mapping multiple mods and should raise an error
    # Pattern matches: letter followed by [number] followed by another [number]
    complex_mods = residue_mods.filter(pl.col("modification").str.contains(r"\]\["))

    if len(complex_mods) > 0:
        complex_mod_list = complex_mods["modification"].to_list()
        raise ValueError(
            f"Found residue modifications with multiple modifications on a single amino acid "
            f"that require mapping multiple mods: {complex_mod_list}. "
            f"These are not supported."
        )

    # Extract the leading amino acid letter from the modification (e.g., "V[242]" -> "V")
    # The modification format is "A[number]" where A is the amino acid
    residue_mods = residue_mods.with_columns(
        pl.col("modification")
        .str.extract(
            r"([A-Z])\[", 1
        )  # Extract the uppercase letter before the opening bracket
        .alias("amino_acid")
    )

    # Append "{amino_acid}" to the UNIMOD encoding
    residue_mods = residue_mods.with_columns(
        (pl.col("amino_acid") + pl.col("proposed_unimod_encoding")).alias(
            "unimod_with_aa"
        )
    )

    # Create dictionary mapping modification to UNIMOD encoding with amino acid
    result_dict = dict(
        zip(
            residue_mods["modification"].to_list(),
            residue_mods["unimod_with_aa"].to_list(),
        )
    )

    return result_dict


def create_n_term_mod_dict(modification_df: pl.DataFrame) -> dict[str, str]:
    """Add ProForma separators so N-terminal labels retain peptide boundaries.

    Args:
        modification_df: Rows containing terminal labels and UNIMOD encodings.

    Returns:
        EncyclopeDIA N-terminal labels mapped to ProForma replacements.
    """
    if modification_df.height == 0:
        return {}

    n_terminal_mods = modification_df.filter(pl.col("modification").str.contains("n"))

    if len(n_terminal_mods) == 0:
        return {}

    # Extract the trailing amino acid letter from the modification (e.g., "n[43]V" -> "V")
    # The modification format is "n[number]A" where A is the amino acid
    n_terminal_mods = n_terminal_mods.with_columns(
        pl.col("modification")
        .str.extract(
            r"\]([A-Z])$", 1
        )  # Extract the uppercase letter after the closing bracket
        .alias("amino_acid")
    )

    # Append "-{amino_acid}" to the UNIMOD encoding
    n_terminal_mods = n_terminal_mods.with_columns(
        (pl.col("proposed_unimod_encoding") + pl.lit("-") + pl.col("amino_acid")).alias(
            "unimod_with_aa"
        )
    )

    # Create dictionary mapping modification to UNIMOD encoding with amino acid
    result_dict = dict(
        zip(
            n_terminal_mods["modification"].to_list(),
            n_terminal_mods["unimod_with_aa"].to_list(),
        )
    )

    return result_dict


def create_c_term_mod_dict(modification_df: pl.DataFrame) -> dict[str, str]:
    """Add ProForma separators so C-terminal labels retain peptide boundaries.

    Args:
        modification_df: Rows containing terminal labels and UNIMOD encodings.

    Returns:
        EncyclopeDIA C-terminal labels mapped to ProForma replacements.

    Raises:
        ValueError: If a C-terminal label also contains an unsupported residue modification.
    """
    if modification_df.height == 0:
        return {}

    c_terminal_mods = modification_df.filter(pl.col("modification").str.contains("c"))

    if len(c_terminal_mods) == 0:
        return {}

    # Check for modifications with residue modifications before C-terminal modifications
    # e.g., "K[170]c[123]" - these require mapping two mods and should raise an error
    complex_mods = c_terminal_mods.filter(
        pl.col("modification").str.contains(r"\[\d+\]c\[")
    )

    if len(complex_mods) > 0:
        complex_mod_list = complex_mods["modification"].to_list()
        raise ValueError(
            f"Found C-terminal modifications with preceding residue modifications "
            f"that require mapping two mods: {complex_mod_list}. "
            f"These are not supported."
        )

    # Extract the leading amino acid letter from the modification (e.g., "Qc[111]" -> "Q")
    # Only handles simple cases where the amino acid is directly before "c["
    c_terminal_mods = c_terminal_mods.with_columns(
        pl.col("modification")
        .str.extract(
            r"([A-Z])c\[", 1
        )  # Extract the uppercase letter directly before "c["
        .alias("amino_acid")
    )

    # Prepend "{amino_acid}-" to the UNIMOD encoding
    c_terminal_mods = c_terminal_mods.with_columns(
        (pl.col("amino_acid") + pl.lit("-") + pl.col("proposed_unimod_encoding")).alias(
            "unimod_with_aa"
        )
    )

    # Create dictionary mapping modification to UNIMOD encoding with amino acid
    result_dict = dict(
        zip(
            c_terminal_mods["modification"].to_list(),
            c_terminal_mods["unimod_with_aa"].to_list(),
        )
    )

    return result_dict


def replace_modifications(
    peptide: str,
    modification_dict: dict[str, str],
    n_term_dict: dict[str, str],
    c_term_dict: dict[str, str],
) -> str:
    """Apply residue and terminal mappings in an order that preserves ProForma syntax.

    Terminal modifications are separated from the peptide by hyphens:
    - N-terminal: n[43]MPEPTIDE -> [UNIMOD:1]-MPEPTIDE
    - C-terminal: PEPTIDEc[45] -> PEPTIDE-[UNIMOD:xxx]

    Args:
        peptide: EncyclopeDIA-annotated peptide string.
        modification_dict: Residue-level replacement mapping.
        n_term_dict: N-terminal replacement mapping with separators.
        c_term_dict: C-terminal replacement mapping with separators.

    Returns:
        Peptide string using ProForma UNIMOD labels.
    """
    result = peptide

    # Apply N-terminal modifications with trailing hyphen
    for old_mod, new_mod in n_term_dict.items():
        result = result.replace(old_mod, new_mod)

    # Apply C-terminal modifications with leading hyphen
    for old_mod, new_mod in c_term_dict.items():
        result = result.replace(old_mod, new_mod)

    # Apply regular residue modifications
    for old_mod, new_mod in modification_dict.items():
        result = result.replace(old_mod, new_mod)

    return result


def create_file_specific_mod_dicts(
    file_name: str,
    pxd009449_ambiguous_modifications_df: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Resolve ambiguous PXD009449 labels from filename context before replacement.

    Filters the ambiguous modifications DataFrame to find rows where the file name
    contains the modification_in_file_name, then creates modification dictionaries
    from those matching rows.

    Args:
        file_name: Parquet filename carrying disambiguating context.
        pxd009449_ambiguous_modifications_df: Override rows with filename patterns.

    Returns:
        File-specific residue, N-terminal, and C-terminal mappings.
    """
    # Filter modifications where the file name contains modification_in_file_name
    # Use map_elements to check if modification_in_file_name is a substring of file_name
    matching_mods = pxd009449_ambiguous_modifications_df.filter(
        pl.col("modification_in_file_name").map_elements(
            lambda x: x in file_name if x is not None else False,
            return_dtype=pl.Boolean,
        )
    )

    if len(matching_mods) == 0:
        # No matching modifications, return empty dicts (will use gold standard)
        return {}, {}, {}

    # Create dictionaries from matching modifications
    mod_dict = create_mod_dict(matching_mods)
    n_term_mod_dict = create_n_term_mod_dict(matching_mods)
    c_term_mod_dict = create_c_term_mod_dict(matching_mods)

    return mod_dict, n_term_mod_dict, c_term_mod_dict


def create_replacement_function(
    modification_dict: dict[str, str],
    n_term_dict: dict[str, str],
    c_term_dict: dict[str, str],
) -> Callable[[str], str]:
    """Bind validated mappings once for efficient Polars element conversion.

    Args:
        modification_dict: Residue-level replacement mapping.
        n_term_dict: N-terminal replacement mapping.
        c_term_dict: C-terminal replacement mapping.

    Returns:
        Callable that translates one peptide string.
    """
    return lambda peptide: replace_modifications(
        peptide, modification_dict, n_term_dict, c_term_dict
    )


def create_unimod_column(
    subfolder: str,
    mod_dict: dict[str, str],
    n_term_mod_dict: dict[str, str],
    c_term_mod_dict: dict[str, str],
    pxd009449_ambiguous_modifications_df: pl.DataFrame,
    sequence_col: str = "peptide",
    modified_sequence_col: str = "modified_peptide",
    drop_old_modifications: bool = False,
) -> None:
    """Rewrite Parquet sequences into the canonical training representation.

    Args:
        subfolder: Dataset tree containing Parquet files.
        mod_dict: Default residue-level mappings.
        n_term_mod_dict: Default N-terminal mappings.
        c_term_mod_dict: Default C-terminal mappings.
        pxd009449_ambiguous_modifications_df: Filename-dependent PXD009449 overrides.
        sequence_col: Column containing unmodified peptide sequences.
        modified_sequence_col: Column containing EncyclopeDIA annotations.
        drop_old_modifications: Whether to remove the source annotation column.
    """
    files = glob.glob(f"{subfolder}/**/*.parquet", recursive=True)
    logger.info(f"Found {len(files)} files to process in {subfolder}")

    for file in tqdm(files, unit="file", desc=f"Processing {subfolder}"):
        # Check if the immediate parent directory (subfolder containing the file) exactly matches PXD009449
        file_dir = os.path.dirname(file)
        parent_dir_name = os.path.basename(file_dir)
        file_is_pxd009449 = parent_dir_name == "PXD009449"
        file_name = os.path.basename(file)  # Extract filename for matching

        df = pl.scan_parquet(file)

        df = df.with_columns(pl.col(sequence_col).alias("unmodified_peptide"))

        # First merge the columns with modified_sequence taking precedence
        df = df.with_columns(
            pl.coalesce([pl.col(modified_sequence_col), pl.col(sequence_col)])
            .cast(pl.String)
            .alias("sequence")
        )

        # Determine which dictionaries to use
        if file_is_pxd009449:
            # Get file-specific dictionaries based on filename matching
            file_mod_dict, file_n_term_dict, file_c_term_dict = (
                create_file_specific_mod_dicts(
                    file_name, pxd009449_ambiguous_modifications_df
                )
            )

            # Merge with gold standard: file-specific takes precedence
            # This ensures we don't overwrite already-processed modifications
            final_mod_dict = {**mod_dict, **file_mod_dict}
            final_n_term_dict = {**n_term_mod_dict, **file_n_term_dict}
            final_c_term_dict = {**c_term_mod_dict, **file_c_term_dict}
        else:
            # Use gold standard dictionaries
            final_mod_dict = mod_dict
            final_n_term_dict = n_term_mod_dict
            final_c_term_dict = c_term_mod_dict

        # Apply the modifications using the appropriate dictionaries
        replacement_func = create_replacement_function(
            final_mod_dict, final_n_term_dict, final_c_term_dict
        )
        df = df.with_columns(
            pl.col("sequence")
            .map_elements(replacement_func, return_dtype=pl.String)
            .alias("sequence")
        )

        if drop_old_modifications:
            # Drop old columns
            df = df.drop([modified_sequence_col])
        if sequence_col != "unmodified_peptide":
            df = df.drop([sequence_col])
        df.collect().write_parquet(file)


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Dataset tree containing parquet files (repeatable)",
        ),
    ],
    gold_standard_mods: Annotated[
        Path,
        typer.Option(
            "--gold-standard-mods",
            help="Gold standard modifications Excel file",
        ),
    ] = DEFAULT_GOLD_STANDARD_MODS,
    ambiguous_mods: Annotated[
        Path,
        typer.Option(
            "--ambiguous-mods",
            help="PXD009449 ambiguous modifications Excel file",
        ),
    ] = DEFAULT_AMBIGUOUS_MODS,
    sequence_col: Annotated[
        str,
        typer.Option("--sequence-col", help="Name of the unmodified sequence column"),
    ] = "unmodified_peptide",
    modified_sequence_col: Annotated[
        str,
        typer.Option(
            "--modified-sequence-col",
            help="Name of the modified sequence column",
        ),
    ] = "modified_peptide",
    drop_old_modifications: Annotated[
        bool,
        typer.Option(
            "--drop-old-modifications",
            help="Drop the source modified-sequence column after labelling",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Translate EncyclopeDIA modifications to UNIMOD in parquet datasets."""
    configure_script_logging(verbose=verbose)

    gold_standard_modifications_df = read_gold_standard_modifications(
        str(gold_standard_mods)
    )
    pxd009449_ambiguous_modifications_df = read_pxd009449_ambiguous_modifications(
        str(ambiguous_mods)
    )

    mod_dict = create_mod_dict(gold_standard_modifications_df)
    n_term_mod_dict = create_n_term_mod_dict(gold_standard_modifications_df)
    c_term_mod_dict = create_c_term_mod_dict(gold_standard_modifications_df)

    for subfolder in input_dir:
        logger.info(f"Processing subfolder: {subfolder}")
        create_unimod_column(
            subfolder=str(subfolder),
            mod_dict=mod_dict,
            n_term_mod_dict=n_term_mod_dict,
            c_term_mod_dict=c_term_mod_dict,
            pxd009449_ambiguous_modifications_df=pxd009449_ambiguous_modifications_df,
            sequence_col=sequence_col,
            modified_sequence_col=modified_sequence_col,
            drop_old_modifications=drop_old_modifications,
        )


if __name__ == "__main__":
    app()
