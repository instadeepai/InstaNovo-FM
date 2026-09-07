from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import typer
from tqdm import tqdm

from instanovo.constants import ANNOTATED_COLUMN
from instanovo.utils.data_handler import SpectrumDataFrame
from scripts.preprocessing.add_acquisition_column import (
    extract_file_name,
    load_acquisitions_from_search_data,
)
from scripts.preprocessing.parquet_io import (
    align_dataframe_to_schema,
    atomic_write_parquet,
    experiment_name_from_path,
)
from scripts.preprocessing.recombine_acfm_random_splits import ACFM_REFERENCE_DTYPES
from scripts.verification.add_usi_column import build_usi_string

app = typer.Typer(help="Convert IPC files to Parquet format")

# Module-level constants to avoid B008 errors
SOURCE_DIR_OPTION = typer.Option(
    None, "--source-dir", "-s", help="Directory to search for IPC files"
)
INPUT_FILE_OPTION = typer.Option(
    None, "--input-file", "-i", help="File listing IPC files to convert"
)
OUTPUT_FILE_OPTION = typer.Option(
    "conversion_errors.txt", "--output", "-o", help="Error log file"
)
MAX_SHARD_SIZE_OPTION = typer.Option(
    500_000, "--max-shard-size", "-m", help="Maximum shard size"
)
LAZY_OPTION = typer.Option(True, "--lazy/--no-lazy", help="Use lazy loading")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
COLUMN_MAPPING_OPTION = typer.Option(
    None, "--column-mapping", "-c", help="JSON string for column mapping"
)
SEARCH_DATA_OPTION = typer.Option(
    None,
    "--search-data",
    help="Search metadata Excel for acquisition lookup (enables ACFM metadata columns)",
)
ADD_USI_OPTION = typer.Option(
    True,
    "--add-usi/--no-usi",
    help="Build USI column during ACFM metadata enrichment (requires --search-data)",
)
INPUT_FILES_ARG = typer.Argument(..., help="Input files listing IPC files to convert")
BATCH_OUTPUT_FILE_OPTION = typer.Option(
    "batch_conversion_errors.txt", "--output", "-o", help="Error log file"
)


def parquet_path_for_ipc_shard(
    ipc_path: str | Path, shard_counter: int, num_shards: int
) -> Path:
    """Return the parquet output path for one IPC shard."""
    ipc_path = Path(ipc_path)
    root = ipc_path.parent
    ipc_name = ipc_path.name
    if num_shards == 1:
        parquet_name = ipc_name.replace(".ipc", ".parquet")
    else:
        stem = ipc_name.replace(".ipc", "")
        parquet_name = f"{stem}_{shard_counter:04d}-{num_shards:04d}.parquet"
    return root / parquet_name


def _add_usi_column(df: pl.DataFrame, parquet_path: Path) -> pl.DataFrame:
    if "scan" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.String).alias("usi"))

    usi_values = [
        build_usi_string(
            str(parquet_path),
            row.get("scan"),
            sequence=None,
            precursor_charge=row.get("precursor_charge"),
        )
        for row in df.iter_rows(named=True)
    ]
    return df.with_columns(pl.Series("usi", usi_values, dtype=pl.String))


def enrich_acfm_metadata(
    df: pl.DataFrame,
    parquet_path: Path,
    acquisition: Optional[str],
    reference_dtypes: Dict[str, pl.DataType] = ACFM_REFERENCE_DTYPES,
    add_usi: bool = True,
) -> pl.DataFrame:
    """Add canonical ``experiment_name``, ``acquisition``, and optionally ``usi``."""
    experiment_name = experiment_name_from_path(str(parquet_path))
    df = df.with_columns(
        pl.lit(experiment_name).alias("experiment_name"),
        pl.lit(acquisition).cast(pl.String).alias("acquisition"),
    )
    if add_usi:
        df = _add_usi_column(df, parquet_path)
    return align_dataframe_to_schema(df, reference_dtypes)


def _prepare_ipc_shard(
    shard: pl.DataFrame, column_mapping: dict[str, str]
) -> pl.DataFrame:
    """Apply the same per-shard transforms as ``SpectrumDataFrame.get_data_shards``.

    Mirrors ``_df_from_ipc`` (the ``modified_sequence`` → annotated-column alias)
    followed by the column rename and dtype casting, so a shard read via a lazy
    slice is equivalent to one produced by the previous full-file path.
    """
    if "modified_sequence" in shard.columns:
        shard = shard.with_columns(pl.col("modified_sequence").alias(ANNOTATED_COLUMN))
    shard = shard.rename(
        {k: v for k, v in column_mapping.items() if k in shard.columns}
    )
    return SpectrumDataFrame._cast_columns(shard)


def convert_ipc_with_metadata(
    ipc_path: str,
    acquisition_map: Dict[tuple[str, str], str],
    column_mapping: Optional[dict[str, str]] = None,
    max_shard_size: int = 500_000,
    verbose: bool = False,
    add_usi: bool = True,
) -> None:
    """Convert one IPC file to sharded parquet with ACFM metadata columns.

    The IPC file is read shard-by-shard through a lazy scan, so peak memory is
    bounded by ``max_shard_size`` rows regardless of the file's total size —
    previously the whole file was materialised (once to count rows and again to
    shard it), which OOMed on large IPC files.
    """
    ipc_path_obj = Path(ipc_path)
    project = ipc_path_obj.parent.name

    lazy_frame = pl.scan_ipc(ipc_path)
    original_file_length = lazy_frame.select(pl.len()).collect().item()
    # Naming denominator preserved from the previous implementation for
    # byte-identical output filenames.
    num_shards = original_file_length // max_shard_size + 1

    first_parquet = parquet_path_for_ipc_shard(ipc_path, 0, num_shards)
    lookup_key = extract_file_name(str(first_parquet))
    acquisition = acquisition_map.get((project, lookup_key))
    if acquisition is None:
        raise ValueError(f"No acquisition in search data for {project}/{lookup_key}")

    column_mapping = column_mapping or {}
    # Number of shards actually written == ceil(len / max), matching the count
    # yielded by the previous get_data_shards path (min 1 for empty files).
    n_shards_to_write = max(
        1, (original_file_length + max_shard_size - 1) // max_shard_size
    )
    if verbose:
        typer.echo(
            f"{ipc_path}: {original_file_length:,} rows -> {n_shards_to_write} shard(s)"
        )

    for shard_counter in range(n_shards_to_write):
        shard = lazy_frame.slice(
            shard_counter * max_shard_size, max_shard_size
        ).collect()
        shard = _prepare_ipc_shard(shard, column_mapping)
        parquet_path = parquet_path_for_ipc_shard(ipc_path, shard_counter, num_shards)
        enriched = enrich_acfm_metadata(
            shard, parquet_path, acquisition, add_usi=add_usi
        )
        atomic_write_parquet(enriched, parquet_path)


def convert_ipc_to_parquet(
    source_dir: Optional[str] = None,
    input_file: Optional[str] = None,
    output_file: str = "unexpected_conversion_errors.txt",
    column_mapping: Optional[dict[str, str]] = None,
    max_shard_size: int = 500_000,
    lazy: bool = True,
    verbose: bool = False,
    search_data_path: Optional[Path] = None,
    add_usi: bool = True,
) -> None:
    """Converts IPC files to Parquet format.

    When *search_data_path* is set, writes ``experiment_name`` and ``acquisition``
    during conversion, and ``usi`` when *add_usi* is true.
    """
    if column_mapping is None:
        column_mapping = {
            "rt": "retention_time",
            "mz": "mz_array",
            "intensity": "intensity_array",
            "peptide": "unmodified_peptide",
        }

    if verbose:
        typer.echo(f"Source directory: {source_dir}")
        typer.echo(f"Input file: {input_file}")
        typer.echo(f"Output error file: {output_file}")
        typer.echo(f"Max shard size: {max_shard_size}")
        typer.echo(f"Lazy loading: {lazy}")
        typer.echo(f"Search data: {search_data_path}")
        typer.echo(f"Add USI: {add_usi}")

    ipc_files = collect_ipc_files(source_dir, input_file)

    if verbose:
        typer.echo(f"Found {len(ipc_files)} IPC files to convert")

    acquisition_map: Optional[Dict[tuple[str, str], str]] = None
    if search_data_path is not None:
        if not search_data_path.is_file():
            raise typer.BadParameter(f"Search data file not found: {search_data_path}")
        acquisition_map = load_acquisitions_from_search_data(str(search_data_path))

    process_ipc_files(
        ipc_files,
        output_file,
        column_mapping,
        max_shard_size,
        lazy,
        verbose,
        acquisition_map,
        add_usi,
    )


def collect_ipc_files(
    source_dir: Optional[str] = None, input_file: Optional[str] = None
) -> list:
    """Collects IPC file paths from a directory or an input file."""
    ipc_files = []
    if source_dir:
        ipc_files.extend(find_ipc_files_in_directory(source_dir))
    elif input_file:
        ipc_files.extend(read_ipc_files_from_file(input_file))
    return ipc_files


def find_ipc_files_in_directory(source_dir: str) -> list:
    """Finds all IPC files in the given directory."""
    ipc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipc"):
                ipc_files.append(os.path.join(root, file))
    return ipc_files


def read_ipc_files_from_file(input_file: str) -> list:
    """Reads IPC file paths from the input file."""
    if not os.path.exists(input_file):
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    ipc_files = []
    with open(input_file, "r") as f:
        for line in f:
            file_path = line.split(":")[0].strip()
            if file_path.endswith(".ipc"):
                ipc_files.append(file_path)
    return ipc_files


def process_ipc_files(
    ipc_files: list,
    output_file: str,
    column_mapping: Optional[dict[str, str]] = None,
    max_shard_size: int = 500_000,
    lazy: bool = True,
    verbose: bool = False,
    acquisition_map: Optional[Dict[tuple[str, str], str]] = None,
    add_usi: bool = True,
) -> None:
    """Processes each IPC file and converts it to Parquet."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=len(ipc_files), desc="Converting IPC to Parquet", unit="file"
    ) as pbar:
        for ipc_path in ipc_files:
            try:
                if verbose:
                    typer.echo(f"Converting: {ipc_path}")

                if acquisition_map is not None:
                    convert_ipc_with_metadata(
                        ipc_path,
                        acquisition_map,
                        column_mapping=column_mapping,
                        max_shard_size=max_shard_size,
                        verbose=verbose,
                        add_usi=add_usi,
                    )
                else:
                    SpectrumDataFrame.load(
                        ipc_path,
                        column_mapping=column_mapping,
                        lazy=lazy,
                        max_shard_size=max_shard_size,
                        load_and_save=True,
                    )
                pbar.update(1)
            except Exception as e:
                log_error(output_file, ipc_path, e)
                if verbose:
                    typer.echo(f"Error converting {ipc_path}: {e}", err=True)


def log_error(output_file: str, ipc_path: str, error: Exception) -> None:
    """Logs errors encountered during file processing."""
    with open(output_file, "a") as log:
        log.write(f"Error processing {ipc_path}: {error}\n")


@app.command()
def convert(
    source_dir: Optional[str] = SOURCE_DIR_OPTION,
    input_file: Optional[str] = INPUT_FILE_OPTION,
    output_file: str = OUTPUT_FILE_OPTION,
    max_shard_size: int = MAX_SHARD_SIZE_OPTION,
    lazy: bool = LAZY_OPTION,
    verbose: bool = VERBOSE_OPTION,
    column_mapping: Optional[str] = COLUMN_MAPPING_OPTION,
    search_data: Optional[Path] = SEARCH_DATA_OPTION,
    add_usi: bool = ADD_USI_OPTION,
) -> None:
    """Convert IPC files to Parquet format."""
    parsed_column_mapping = None
    if column_mapping:
        try:
            parsed_column_mapping = json.loads(column_mapping)
        except json.JSONDecodeError:
            typer.echo("Error: Invalid JSON format for column mapping", err=True)
            raise typer.Exit(1)

    if not parsed_column_mapping:
        parsed_column_mapping = {
            "rt": "retention_time",
            "mz": "mz_array",
            "intensity": "intensity_array",
        }

    if not source_dir and not input_file:
        typer.echo("Error: Must provide either --source-dir or --input-file", err=True)
        raise typer.Exit(1)

    convert_ipc_to_parquet(
        source_dir=source_dir,
        input_file=input_file,
        output_file=output_file,
        column_mapping=parsed_column_mapping,
        max_shard_size=max_shard_size,
        lazy=lazy,
        verbose=verbose,
        search_data_path=search_data,
        add_usi=add_usi,
    )


@app.command()
def batch_convert(
    input_files: list[str] = INPUT_FILES_ARG,
    output_file: str = BATCH_OUTPUT_FILE_OPTION,
    max_shard_size: int = MAX_SHARD_SIZE_OPTION,
    lazy: bool = LAZY_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Convert IPC files listed in multiple input files."""
    column_mapping = {
        "rt": "retention_time",
        "mz": "mz_array",
        "intensity": "intensity_array",
    }

    for input_file in input_files:
        if not os.path.exists(input_file):
            typer.echo(
                f"Warning: Input file '{input_file}' does not exist, skipping...",
                err=True,
            )
            continue

        typer.echo(f"Processing input file: {input_file}")
        convert_ipc_to_parquet(
            input_file=input_file,
            output_file=output_file,
            column_mapping=column_mapping,
            max_shard_size=max_shard_size,
            lazy=lazy,
            verbose=verbose,
        )


def main() -> None:
    """Entry point for the script to convert IPC files to Parquet format."""
    column_mapping = {
        "rt": "retention_time",
        "mz": "mz_array",
        "intensity": "intensity_array",
    }

    input_files = [
        "preprocessing/outputs/missing_files_lcfm.txt",
        "preprocessing/outputs/missing_files_acfm.txt",
    ]

    for input_file in input_files:
        if os.path.exists(input_file):
            typer.echo(f"Processing: {input_file}")
            convert_ipc_to_parquet(
                input_file=input_file,
                column_mapping=column_mapping,
            )
        else:
            typer.echo(f"Warning: Input file '{input_file}' does not exist", err=True)


if __name__ == "__main__":
    app()
