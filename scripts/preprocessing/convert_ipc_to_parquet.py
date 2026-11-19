"""Convert IPC spectra into schema-aligned, optionally enriched Parquet shards.

Run this after validating IPC inputs and before Parquet-only preprocessing.
Large files are sliced lazily to bound memory, while optional search data adds
ACFM acquisition and USI metadata during conversion.

CLI::

    uv run python -m scripts.preprocessing.convert_ipc_to_parquet --help
    uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-dir <data-root>/acfm
    uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-file missing_files_lcfm.txt --input-file missing_files_acfm.txt

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import polars as pl
import typer
from tqdm import tqdm

from instanovo.constants import ANNOTATED_COLUMN
from instanovo.utils.data_handler import SpectrumDataFrame
from scripts.logging_setup import configure_script_logging
from scripts.preprocessing.add_acquisition_column import (
    extract_file_name,
    load_acquisitions_from_search_data,
)
from scripts.preprocessing.parquet_io import (
    align_dataframe_to_schema,
    atomic_write_parquet,
    experiment_name_from_path,
)
from scripts.verification.add_usi_column import build_usi_string

logger = logging.getLogger(__name__)

# Unlabelled ACFM columns, plus the pre-inference isolation column kept by
# ``infer_isolation_target.py``.
ACFM_REFERENCE_DTYPES: Dict[str, pl.DataType] = {
    "usi": pl.String,
    "index": pl.Int64,
    "scan": pl.String,
    "header": pl.String,
    "retention_time": pl.Float64,
    "frag_type": pl.String,
    "acquisition": pl.String,
    "collision_energy": pl.Float64,
    "isolation_target": pl.Float64,
    "precursor_mz": pl.Float64,
    "precursor_charge": pl.Int64,
    "precursor_intensity": pl.Float64,
    "lower_offset": pl.Float64,
    "upper_offset": pl.Float64,
    "mz_array": pl.List(pl.Float64),
    "intensity_array": pl.List(pl.Float32),
    "scale_factor": pl.Float32,
    "experiment_name": pl.String,
    "isolation_target_old": pl.Float64,
}

app = typer.Typer(
    help="Convert IPC files to Parquet format",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_COLUMN_MAPPING: dict[str, str] = {
    "rt": "retention_time",
    "mz": "mz_array",
    "intensity": "intensity_array",
    "peptide": "unmodified_peptide",
}


def resolve_column_mapping(
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Merge user overrides onto the shared default mapping without dropping keys."""
    mapping = dict(DEFAULT_COLUMN_MAPPING)
    if overrides:
        mapping.update(overrides)
    return mapping


def parquet_path_for_ipc_shard(
    ipc_path: str | Path, shard_counter: int, num_shards: int
) -> Path:
    """Keep whole-file and sharded output names compatible with downstream grouping.

    Args:
        ipc_path: Source IPC path.
        shard_counter: Zero-based shard index.
        num_shards: Denominator encoded in sharded filenames.

    Returns:
        Parquet destination for the requested shard.
    """
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
    """Keep metadata enrichment usable when source IPC files omit scan identifiers."""
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
    """Make converted ACFM shards immediately compatible with the reference schema.

    Args:
        df: Converted IPC shard to enrich.
        parquet_path: Output path used to derive stable experiment metadata.
        acquisition: Acquisition type resolved from search data.
        reference_dtypes: Canonical ACFM schema.
        add_usi: Whether to construct universal spectrum identifiers.

    Returns:
        Enriched shard aligned to the reference schema.
    """
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
    """Preserve ``SpectrumDataFrame`` conversion semantics while reading IPC lazily."""
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
    """Bound peak memory while producing complete, schema-aligned ACFM shards.

    The IPC file is read shard-by-shard through a lazy scan, so peak memory is
    bounded by ``max_shard_size`` rows regardless of total size.

    Args:
        ipc_path: IPC file to convert.
        acquisition_map: Project and filename keys mapped to acquisition type.
        column_mapping: Source columns mapped to canonical names.
        max_shard_size: Maximum rows materialised per output shard.
        verbose: Whether to print shard details.
        add_usi: Whether to construct universal spectrum identifiers.

    Raises:
        ValueError: If search data has no acquisition for the IPC file.
    """
    ipc_path_obj = Path(ipc_path)
    project = ipc_path_obj.parent.name

    lazy_frame = pl.scan_ipc(ipc_path)
    original_file_length = lazy_frame.select(pl.len()).collect().item()
    if original_file_length == 0:
        raise ValueError(f"IPC file is empty: {ipc_path}")

    # ceil(len / max). Filename denominator must match files written so
    # check_conversion accepts exact multiples of max_shard_size.
    n_shards = (original_file_length + max_shard_size - 1) // max_shard_size

    first_parquet = parquet_path_for_ipc_shard(ipc_path, 0, n_shards)
    lookup_key = extract_file_name(str(first_parquet))
    acquisition = acquisition_map.get((project, lookup_key))
    if acquisition is None:
        raise ValueError(f"No acquisition in search data for {project}/{lookup_key}")

    column_mapping = column_mapping or {}
    if verbose:
        logger.debug(
            f"{ipc_path}: {original_file_length:,} rows -> {n_shards} shard(s)"
        )

    for shard_counter in range(n_shards):
        shard = lazy_frame.slice(
            shard_counter * max_shard_size, max_shard_size
        ).collect()
        shard = _prepare_ipc_shard(shard, column_mapping)
        parquet_path = parquet_path_for_ipc_shard(ipc_path, shard_counter, n_shards)
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
    """Produce Parquet inputs that downstream preprocessing can consume efficiently.

    When *search_data_path* is set, writes ``experiment_name`` and ``acquisition``
    during conversion, and ``usi`` when *add_usi* is true.

    Args:
        source_dir: Directory tree containing IPC files.
        input_file: Text report listing IPC files to convert.
        output_file: Destination for per-file conversion errors.
        column_mapping: Source columns mapped to canonical names.
        max_shard_size: Maximum rows per Parquet shard.
        lazy: Whether the standard converter should use lazy loading.
        verbose: Whether to print processing details.
        search_data_path: Optional workbook enabling ACFM metadata enrichment.
        add_usi: Whether to add USIs when metadata enrichment is enabled.

    Raises:
        typer.BadParameter: If the requested search-data workbook does not exist.
    """
    if column_mapping is None:
        column_mapping = resolve_column_mapping()

    logger.debug(f"Source directory: {source_dir}")
    logger.debug(f"Input file: {input_file}")
    logger.debug(f"Output error file: {output_file}")
    logger.debug(f"Max shard size: {max_shard_size}")
    logger.debug(f"Lazy loading: {lazy}")
    logger.debug(f"Search data: {search_data_path}")
    logger.debug(f"Add USI: {add_usi}")

    ipc_files = collect_ipc_files(source_dir, input_file)

    logger.debug(f"Found {len(ipc_files)} IPC files to convert")

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
    """Accept either discovery or a prior report as the source of conversion work.

    Args:
        source_dir: Optional directory tree containing IPC files.
        input_file: Optional text file listing IPC paths.

    Returns:
        IPC paths selected for conversion.
    """
    ipc_files = []
    if source_dir:
        ipc_files.extend(find_ipc_files_in_directory(source_dir))
    elif input_file:
        ipc_files.extend(read_ipc_files_from_file(input_file))
    return ipc_files


def find_ipc_files_in_directory(source_dir: str) -> list:
    """Discover every IPC input when no targeted conversion report is available.

    Args:
        source_dir: Directory tree to search.

    Returns:
        IPC paths found beneath the source directory.
    """
    ipc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipc"):
                ipc_files.append(os.path.join(root, file))
    return ipc_files


def read_ipc_files_from_file(input_file: str) -> list:
    """Turn a validation report into a targeted IPC conversion queue.

    Args:
        input_file: Text file whose lines begin with IPC paths.

    Returns:
        IPC paths parsed from the report.

    Raises:
        typer.Exit: If the report does not exist.
    """
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
    """Continue batch conversion after individual failures and record each error.

    Args:
        ipc_files: IPC paths selected for conversion.
        output_file: Destination for conversion errors.
        column_mapping: Source columns mapped to canonical names.
        max_shard_size: Maximum rows per Parquet shard.
        lazy: Whether the standard converter should use lazy loading.
        verbose: Whether to print per-file details.
        acquisition_map: Optional metadata lookup enabling enriched conversion.
        add_usi: Whether to add USIs during enriched conversion.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=len(ipc_files), desc="Converting IPC to Parquet", unit="file"
    ) as pbar:
        for ipc_path in ipc_files:
            try:
                if verbose:
                    logger.debug(f"Converting: {ipc_path}")

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
                    if pl.scan_ipc(ipc_path).select(pl.len()).collect().item() == 0:
                        raise ValueError(f"IPC file is empty: {ipc_path}")
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
                    logger.error(f"Error converting {ipc_path}: {e}")


def log_error(output_file: str, ipc_path: str, error: Exception) -> None:
    """Preserve failed paths so conversion can be retried without rescanning.

    Args:
        output_file: Error log destination.
        ipc_path: IPC file that failed.
        error: Exception raised during conversion.
    """
    with open(output_file, "a") as log:
        log.write(f"Error processing {ipc_path}: {error}\n")


@app.command()
def main(
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory to search for IPC files",
        ),
    ] = None,
    input_file: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--input-file",
            help="Text file listing IPC paths to convert (repeatable)",
        ),
    ] = None,
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Error log file",
        ),
    ] = Path("conversion_errors.txt"),
    max_shard_size: Annotated[
        int,
        typer.Option("--max-shard-size", help="Maximum rows per output shard"),
    ] = 500_000,
    lazy: Annotated[
        bool,
        typer.Option("--lazy/--no-lazy", help="Use lazy loading"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
    column_mapping: Annotated[
        Optional[str],
        typer.Option("--column-mapping", help="JSON string for column mapping"),
    ] = None,
    search_data: Annotated[
        Optional[Path],
        typer.Option(
            "--search-data",
            help=(
                "Optional search-data Excel (project, raw-filename file path, "
                "acquisition) for ACFM metadata enrichment; "
                "typically data/search_data.xlsx"
            ),
        ),
    ] = None,
    add_usi: Annotated[
        bool,
        typer.Option(
            "--add-usi/--no-usi",
            help="Build USI column during ACFM metadata enrichment (requires --search-data)",
        ),
    ] = True,
) -> None:
    """Convert IPC files to Parquet from a directory and/or path lists."""
    configure_script_logging(verbose=verbose)

    parsed_overrides = None
    if column_mapping:
        try:
            parsed_overrides = json.loads(column_mapping)
        except json.JSONDecodeError:
            typer.echo("Error: Invalid JSON format for column mapping", err=True)
            raise typer.Exit(1)

    parsed_column_mapping = resolve_column_mapping(parsed_overrides)

    input_files = input_file or []
    if input_dir is None and not input_files:
        typer.echo("Error: Must provide either --input-dir or --input-file", err=True)
        raise typer.Exit(1)

    if input_dir is not None:
        convert_ipc_to_parquet(
            source_dir=str(input_dir),
            input_file=None,
            output_file=str(output_file),
            column_mapping=parsed_column_mapping,
            max_shard_size=max_shard_size,
            lazy=lazy,
            verbose=verbose,
            search_data_path=search_data,
            add_usi=add_usi,
        )

    for path in input_files:
        if not path.exists():
            logger.warning(f"Input file '{path}' does not exist, skipping...")
            continue
        logger.info(f"Processing input file: {path}")
        convert_ipc_to_parquet(
            source_dir=None,
            input_file=str(path),
            output_file=str(output_file),
            column_mapping=parsed_column_mapping,
            max_shard_size=max_shard_size,
            lazy=lazy,
            verbose=verbose,
            search_data_path=search_data,
            add_usi=add_usi,
        )


if __name__ == "__main__":
    app()
