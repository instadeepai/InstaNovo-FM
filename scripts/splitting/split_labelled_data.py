r"""Split labelled MS/MS spectra into train/test/valid sets without peptide leakage.

A persistent peptide registry (HuggingFace or a local directory) records each
peptide's split so the same sequence never appears in both train and test,
including across later datasets. Peptides are normalised (UNIMOD tags stripped,
I→L) before lookup; new ones are assigned toward 80/10/10 and written back.

Modes:
    both (default) — update registry + split files
    update-splits  — update and save registry only
    split-only     — split files using existing registry

If *unmodified_peptide* is missing from an input file, it is filled from *sequence*
by removing bracketed segments (e.g. ``[UNIMOD:123]``) and all hyphen (``-``)
characters.

CLI::

    uv run python -m scripts.splitting.split_labelled_data --help
    uv run python -m scripts.splitting.split_labelled_data split --input-dir lcfm --output-dir splits
    uv run python -m scripts.splitting.split_labelled_data split -i lcfm --output-dir splits --mode split-only
    uv run python -m scripts.splitting.split_labelled_data split -i data --output-dir splits \\
        --column-remap '{"legacy_peptide":"unmodified_peptide"}'
    uv run python -m scripts.splitting.split_labelled_data batch dir1 dir2 --output-dir splits/
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import polars as pl
import typer
from huggingface_hub import HfApi, hf_hub_download

from scripts.logging_setup import configure_script_logging

app = typer.Typer(
    help="Split labelled parquet into peptide-disjoint train/test/valid",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)

# ── Quality filter config ────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityFilterConfig:
    """Keeps the spectrum quality cutoffs in one immutable place.

    The same thresholds have to apply when collecting peptides and when writing
    rows.

    Args:
        max_retention_time: Maximum retention time in seconds.
        max_lower_offset: Maximum lower offset in seconds.
        min_precursor_charge: Minimum precursor charge.
        max_precursor_charge: Maximum precursor charge.
        max_precursor_mz: Maximum precursor m/z.
    """

    max_retention_time: float = 10800.0
    max_lower_offset: float = 300.0
    min_precursor_charge: int = 0
    max_precursor_charge: int = 7
    max_precursor_mz: float = 2000.0


QUALITY_FILTERS = QualityFilterConfig()

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
HF_REPO_ID = "InstaDeepAI/InstaNovo"
REGISTRY_FILENAME = "peptide_registry.parquet"
REGISTRY_ENV_TOKEN = "INSTANOVO_HF_TOKEN"
UNIMOD_PATTERN = re.compile(r"\[UNIMOD:\d+\]")
_SPLITS = ("train", "test", "valid")
DESIRED_SPLIT_PROPORTIONS: Dict[str, float] = {
    "train": 0.8,
    "test": 0.1,
    "valid": 0.1,
}

REFERENCE_SCHEMA: Dict[str, pl.DataType] = {
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
    "unmodified_peptide": pl.String,
    "sequence": pl.String,
    "normalised_peptide": pl.String,
}

# Input parquet columns may differ from REFERENCE_SCHEMA names (source → canonical).
# Example: {"legacy_peptide": "unmodified_peptide", "rt": "retention_time"}
REFERENCE_SCHEMA_COLUMN_REMAP: Dict[str, str] = {}

# ── CLI options ──────────────────────────────────────────────────────────────


class Mode(str, Enum):
    """Separate cheap registry updates from expensive split-file writes.

    Args:
        UPDATE_SPLITS: Assign any new peptides and save the registry, writing no
            split files.
        SPLIT_ONLY: Write split files against the existing registry, leaving it
            untouched.
        BOTH: Update the registry, then split using it.
    """

    UPDATE_SPLITS = "update-splits"
    SPLIT_ONLY = "split-only"
    BOTH = "both"


INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-i",
    help="Directory of labelled parquet files to split",
)
OUTPUT_DIR_OPTION = typer.Option(
    ...,
    "--output-dir",
    help="Destination for the registry and split shards",
)
ROWS_PER_FILE_OPTION = typer.Option(
    400_000,
    "--rows-per-file",
    help="Rows per output shard",
)
REGISTRY_DIR_OPTION = typer.Option(
    None,
    "--registry-dir",
    help="Local registry directory; omit to download from HuggingFace",
)
MODE_OPTION = typer.Option(
    Mode.BOTH,
    "--mode",
    "-m",
    help="update-splits, split-only, or both",
)
UPLOAD_REGISTRY_OPTION = typer.Option(
    False,
    "--upload-registry-to-hf",
    help="Publish the updated registry to the shared HuggingFace repo",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging")
INPUT_DIRS_ARG = typer.Argument(..., help="Input directories to process as one corpus")
COLUMN_REMAP_OPTION = typer.Option(
    None,
    "--column-remap",
    help=(
        "JSON object of source→canonical column names, e.g. "
        '\'{"legacy_peptide":"unmodified_peptide"}\'. '
        "Merged on top of REFERENCE_SCHEMA_COLUMN_REMAP (CLI wins on duplicate keys)."
    ),
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_column_remap_json(raw: Optional[str]) -> Dict[str, str]:
    """Validate ``--column-remap`` early, so a typo fails before a long run starts.

    Args:
        raw: JSON object mapping source to canonical column names; ``None`` or
            blank means no remapping.

    Returns:
        The parsed mapping, empty when nothing was supplied.

    Raises:
        ValueError: If the value is not valid JSON, is not an object, or holds
            non-string keys or values.
    """
    if raw is None or not str(raw).strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --column-remap: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            "--column-remap must be a JSON object (mapping strings to strings)"
        )
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                "--column-remap requires string keys and string values only"
            )
        out[k] = v
    return out


def effective_column_rename(cli_json: Optional[str]) -> Dict[str, str]:
    """Let a caller override the built-in column mapping without editing the module.

    Args:
        cli_json: Raw ``--column-remap`` JSON, if any.

    Returns:
        :data:`REFERENCE_SCHEMA_COLUMN_REMAP` merged with the CLI mapping, the
        CLI winning on duplicate keys.
    """
    cli_part = parse_column_remap_json(cli_json)
    return {**REFERENCE_SCHEMA_COLUMN_REMAP, **cli_part}


def _fmt(seconds: float) -> str:
    """Render durations as h:mm:ss so multi-hour progress logs stay readable."""
    return str(timedelta(seconds=int(seconds)))


def _resolved_column_rename(
    column_rename: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Distinguish "caller passed no mapping" from "caller asked for none"."""
    return column_rename if column_rename is not None else REFERENCE_SCHEMA_COLUMN_REMAP


def _build_peptide_to_split(split_lookup: Dict[str, Set[str]]) -> Dict[str, str]:
    """Invert the per-split peptide sets so each row can be labelled with one lookup."""
    peptide_to_split: Dict[str, str] = {}
    for split_name, peptides in split_lookup.items():
        for peptide in peptides:
            peptide_to_split[peptide] = split_name
    return peptide_to_split


def _empty_split_buffer_state() -> (
    Tuple[
        Dict[str, List[pl.DataFrame]],
        Dict[str, int],
        Dict[str, int],
    ]
):
    """Start every split with the same buffer, counter and size triple, so writing stays uniform."""
    split_buffers: Dict[str, List[pl.DataFrame]] = {
        split_name: [] for split_name in _SPLITS
    }
    file_counters: Dict[str, int] = {split_name: 0 for split_name in _SPLITS}
    buffer_sizes: Dict[str, int] = {split_name: 0 for split_name in _SPLITS}
    return split_buffers, file_counters, buffer_sizes


def _log_splitting_progress_if_due(
    file_index: int, total_files: int, start_time: float
) -> None:
    """Report throughput and ETA occasionally, so a multi-hour run is observable but not noisy."""
    if file_index % 100 != 0:
        return
    elapsed = time.time() - start_time
    rate = (file_index + 1) / elapsed if elapsed > 0 else 0
    eta = (total_files - file_index - 1) / rate if rate > 0 else 0
    logger.info(
        f"Splitting: {file_index + 1}/{total_files} "
        f"({(file_index + 1) / total_files:.1%}) "
        f"elapsed={_fmt(elapsed)} eta={_fmt(eta)}"
    )


def find_parquet_files(input_dirs: List[str]) -> List[str]:
    """Collect the input corpus in a fixed order, so a seeded run is reproducible.

    Args:
        input_dirs: Directories to walk recursively.

    Returns:
        Every ``.parquet`` path found, sorted.
    """
    files: List[str] = []
    for d in input_dirs:
        for root, _, names in os.walk(d):
            for name in names:
                if name.endswith(".parquet"):
                    files.append(os.path.join(root, name))
    files.sort()
    return files


def _sequence_has_glyco_mods() -> pl.Expr:
    """Flag internal ``[IN:<int>]`` modification tokens, whose rows the quality filter excludes."""
    return (
        pl.col("sequence")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.contains(r"\[[Ii][Nn]:\d+\]")
    )


def _unmodified_peptide_from_sequence_expr() -> pl.Expr:
    """Recover the bare amino-acid string, since leakage is judged on the unmodified peptide."""
    return (
        pl.col("sequence")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"\([^\)]*\)", "")
        .str.replace_all(r"\[[^\]]*\]", "")
        .str.replace_all("-", "")
    )


def _ensure_unmodified_peptide(df: pl.DataFrame) -> pl.DataFrame:
    """Accept inputs that only carry *sequence*, rather than failing the registry lookup."""
    if "unmodified_peptide" in df.columns:
        return df
    if "sequence" not in df.columns:
        raise ValueError(
            "Cannot derive unmodified_peptide: column absent and 'sequence' not found"
        )
    logger.debug(
        "Deriving unmodified_peptide from sequence "
        "(stripping bracketed modifications and hyphens)"
    )
    return df.with_columns(
        _unmodified_peptide_from_sequence_expr().alias("unmodified_peptide")
    )


def _ensure_unmodified_peptide_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy counterpart to :func:`_ensure_unmodified_peptide`, for the scan-only peptide pass."""
    names = set(lf.collect_schema().names())
    if "unmodified_peptide" in names:
        return lf
    if "sequence" not in names:
        raise ValueError(
            "Cannot derive unmodified_peptide: column absent and 'sequence' not found"
        )
    return lf.with_columns(
        _unmodified_peptide_from_sequence_expr().alias("unmodified_peptide")
    )


def _apply_schema_column_rename(
    df: pl.DataFrame, column_rename: Dict[str, str]
) -> pl.DataFrame:
    """Map legacy column names onto the canonical schema, never clobbering a real target column."""
    if not column_rename:
        return df
    names = set(df.columns)
    for src, dst in column_rename.items():
        if src in names and dst not in names:
            df = df.rename({src: dst})
            names.remove(src)
            names.add(dst)
    return df


def _apply_schema_column_rename_lazy(
    lf: pl.LazyFrame, column_rename: Dict[str, str]
) -> pl.LazyFrame:
    """Lazy counterpart to :func:`_apply_schema_column_rename`, for the scan-only peptide pass."""
    if not column_rename:
        return lf
    names = set(lf.collect_schema().names())
    for src, dst in column_rename.items():
        if src in names and dst not in names:
            lf = lf.rename({src: dst})
            names.remove(src)
            names.add(dst)
    return lf


def _nullable_filter(col: str, op: str, threshold: float) -> pl.Expr:
    """Let nulls pass, so a column absent from the source data cannot drop every row."""
    if op == "<=":
        cond = pl.col(col) <= threshold
    elif op == ">=":
        cond = pl.col(col) >= threshold
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return pl.col(col).is_null() | cond


def filter_spectra(df: pl.DataFrame) -> pl.DataFrame:
    """Drop spectra the model cannot use, using the same criteria as the peptide pass.

    Null values in any filter column are treated as passing, so a column absent
    from the source data and null-filled during schema normalisation does not
    wipe out the file.

    Args:
        df: Spectra to filter, already normalised to the reference schema.

    Returns:
        Only the rows passing every :data:`QUALITY_FILTERS` criterion and free
        of glyco modifications.
    """
    return df.filter(
        _nullable_filter("retention_time", "<=", QUALITY_FILTERS.max_retention_time)
        & _nullable_filter("lower_offset", "<=", QUALITY_FILTERS.max_lower_offset)
        & _nullable_filter(
            "precursor_charge", ">=", QUALITY_FILTERS.min_precursor_charge
        )
        & _nullable_filter(
            "precursor_charge", "<=", QUALITY_FILTERS.max_precursor_charge
        )
        & _nullable_filter("precursor_mz", "<=", QUALITY_FILTERS.max_precursor_mz)
        & ~_sequence_has_glyco_mods()
    )


def _add_missing_schema_columns(
    df: pl.DataFrame, schema: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """Give every output the same columns, so shards from differing sources concatenate."""
    missing = [
        pl.lit(None).cast(dtype).alias(name)
        for name, dtype in schema.items()
        if name not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)
    return df


def _add_missing_schema_columns_lazy(
    lf: pl.LazyFrame, schema: Dict[str, pl.DataType]
) -> pl.LazyFrame:
    """Lazy counterpart to :func:`_add_missing_schema_columns`, for the scan-only peptide pass."""
    names = set(lf.collect_schema().names())
    missing = [
        pl.lit(None).cast(dtype).alias(name)
        for name, dtype in schema.items()
        if name not in names
    ]
    if missing:
        lf = lf.with_columns(missing)
    return lf


def normalise_dataframe_schema(
    df: pl.DataFrame,
    schema: Dict[str, pl.DataType],
    *,
    column_rename: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Bring a heterogeneous input file onto the canonical schema so its rows can be pooled.

    Args:
        df: Frame to normalise.
        schema: Canonical column names and dtypes to conform to.
        column_rename: Input-to-canonical column mapping. Omit it (leave
            ``None``) when the caller has already renamed and derived columns
            itself, so this step only fills gaps and reorders; see
            :data:`REFERENCE_SCHEMA_COLUMN_REMAP` for the pipeline default.

    Returns:
        The frame with exactly *schema*'s columns, in its order.
    """
    rename_map = column_rename if column_rename is not None else {}
    df = _apply_schema_column_rename(df, rename_map)
    df = _add_missing_schema_columns(df, schema)
    return df.select(list(schema.keys()))


def normalise_lazyframe_schema(
    lf: pl.LazyFrame,
    schema: Dict[str, pl.DataType],
    *,
    column_rename: Optional[Dict[str, str]] = None,
) -> pl.LazyFrame:
    """Normalise without materialising, so the peptide-collection pass stays a scan.

    Args:
        lf: LazyFrame to normalise.
        schema: Canonical column names and dtypes to conform to.
        column_rename: Input-to-canonical column mapping; ``None`` skips renaming.

    Returns:
        A plan selecting exactly *schema*'s columns, in its order.
    """
    rename_map = column_rename if column_rename is not None else {}
    lf = _apply_schema_column_rename_lazy(lf, rename_map)
    lf = _add_missing_schema_columns_lazy(lf, schema)
    return lf.select(list(schema.keys()))


# ── Registry ─────────────────────────────────────────────────────────────────


def load_peptide_registry(
    registry_dir: Optional[str] = None,
) -> Tuple[pl.DataFrame, Dict[str, Set[str]]]:
    """Recover the split decisions made by previous runs, which is what prevents leakage.

    Only the registry file is fetched from HuggingFace, never a full snapshot.

    Args:
        registry_dir: Local directory holding the registry parquet. When
            omitted, the registry is downloaded from HuggingFace.

    Returns:
        The raw registry frame plus a ``{train/test/valid: peptide set}`` lookup
        ready for membership tests.

    Raises:
        FileNotFoundError: If *registry_dir* is given but holds no registry
            file, rather than silently splitting against an empty registry.
    """
    if registry_dir is not None:
        registry_path = Path(registry_dir) / REGISTRY_FILENAME
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Registry not found at {registry_path}. "
                "Provide a valid --registry-dir or omit to download from HF."
            )
        logger.info(f"Loading registry from {registry_path}")
    else:
        logger.info(f"Downloading {REGISTRY_FILENAME} from HuggingFace: {HF_REPO_ID}")
        # hf_hub_download fetches exactly this file. snapshot_download would fetch the
        # whole repo, which once the corpus is published is ~1 TB to read ~60 MB.
        registry_path = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=REGISTRY_FILENAME,
                repo_type="dataset",
                token=os.getenv(REGISTRY_ENV_TOKEN),
            )
        )

    registry_df = pl.read_parquet(registry_path)
    logger.info(f"Loaded registry with {len(registry_df):,} peptides")

    mapped = registry_df.with_columns(
        pl.when(pl.col("split") == "validation")
        .then(pl.lit("valid"))
        .otherwise(pl.col("split"))
        .alias("split")
    )

    existing_splits: Dict[str, Set[str]] = {split_name: set() for split_name in _SPLITS}
    for (split_name,), group in mapped.group_by("split"):
        if split_name in existing_splits:
            existing_splits[split_name] = set(group["peptide"].to_list())

    for split_name in _SPLITS:
        logger.info(f"  {split_name}: {len(existing_splits[split_name]):,} peptides")

    return registry_df, existing_splits


def save_registry(
    existing_splits: Dict[str, Set[str]],
    output_path: Path,
    upload_to_hf: bool = False,
    registry_changed: bool = True,
) -> None:
    """Persist the split decisions so future runs and datasets inherit them.

    Written with the HuggingFace ``validation`` label for schema compatibility.
    An upload failure is logged rather than raised.

    Args:
        existing_splits: Peptide sets per split, as held in memory.
        output_path: Local parquet path to write.
        upload_to_hf: Publish the registry to the shared dataset repo.
        registry_changed: Whether anything was actually assigned this run;
            upload is skipped when false, since the remote is already in sync.
    """
    all_peptides: List[str] = []
    all_split_labels: List[str] = []
    for split_name, peptides in existing_splits.items():
        hf_split = "validation" if split_name == "valid" else split_name
        sorted_peptides = sorted(peptides)
        all_peptides.extend(sorted_peptides)
        all_split_labels.extend([hf_split] * len(sorted_peptides))

    registry_df = pl.DataFrame({"peptide": all_peptides, "split": all_split_labels})
    registry_df.write_parquet(output_path)
    logger.info(f"Saved registry ({len(registry_df):,} peptides) to {output_path}")

    if upload_to_hf and not registry_changed:
        logger.info(
            "Skipping HuggingFace registry upload (no new peptides; remote already in sync)"
        )
        return

    if upload_to_hf:
        logger.info(f"Uploading registry to {HF_REPO_ID}")
        try:
            HfApi().upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=REGISTRY_FILENAME,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=os.getenv(REGISTRY_ENV_TOKEN),
            )
            logger.info("Upload successful")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.info(f"Local file preserved at {output_path}")


# ── Peptide collection ───────────────────────────────────────────────────────


def collect_unique_peptides(
    parquet_files: List[str],
    *,
    column_rename: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """Learn which peptides a dataset contains before any row is written.

    Runs entirely on lazy scans reading only the columns the quality filter and
    peptide derivation need, so the whole corpus can be surveyed cheaply. The
    filter must match :func:`filter_spectra` exactly, or the split pass will
    encounter peptides this pass never registered. Isoleucine is folded to
    leucine because the two are indistinguishable by mass.

    Args:
        parquet_files: Input files to scan.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.

    Returns:
        The set of normalised peptides surviving the quality filter.
    """
    all_peptides: Set[str] = set()
    start_time = time.time()
    total_parquet_files = len(parquet_files)
    rename_map = (
        REFERENCE_SCHEMA_COLUMN_REMAP if column_rename is None else column_rename
    )
    select_cols = [
        "unmodified_peptide",
        "sequence",
        "retention_time",
        "lower_offset",
        "precursor_charge",
        "precursor_mz",
    ]

    for file_index, file_path in enumerate(parquet_files):
        if file_index % 100 == 0:
            elapsed = time.time() - start_time
            rate = (file_index + 1) / elapsed if elapsed > 0 else 0
            eta = (total_parquet_files - file_index - 1) / rate if rate > 0 else 0
            logger.info(
                f"Collecting peptides: {file_index + 1}/{total_parquet_files} "
                f"({(file_index + 1) / total_parquet_files:.1%}) "
                f"elapsed={_fmt(elapsed)} eta={_fmt(eta)}"
            )

        lf = pl.scan_parquet(file_path)
        lf = _apply_schema_column_rename_lazy(lf, rename_map)
        lf = _ensure_unmodified_peptide_lazy(lf)
        lf = normalise_lazyframe_schema(lf, REFERENCE_SCHEMA, column_rename=None)
        peptides = (
            lf.select(select_cols)
            .filter(
                _nullable_filter(
                    "retention_time", "<=", QUALITY_FILTERS.max_retention_time
                )
                & _nullable_filter(
                    "lower_offset", "<=", QUALITY_FILTERS.max_lower_offset
                )
                & _nullable_filter(
                    "precursor_charge", ">=", QUALITY_FILTERS.min_precursor_charge
                )
                & _nullable_filter(
                    "precursor_charge", "<=", QUALITY_FILTERS.max_precursor_charge
                )
                & _nullable_filter(
                    "precursor_mz", "<=", QUALITY_FILTERS.max_precursor_mz
                )
                & ~_sequence_has_glyco_mods()
            )
            .select(
                pl.col("unmodified_peptide")
                .str.replace_all("I", "L")
                .alias("normalised_peptide")
            )
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        all_peptides.update(peptides)

    return all_peptides


# ── Split assignment ─────────────────────────────────────────────────────────


def assign_new_peptides(
    dataset_peptides: Set[str],
    existing_splits: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Place new peptides so this dataset approaches 80/10/10 despite prior assignments.

    Peptides already in the registry cannot be moved without creating leakage,
    so a dataset can arrive with a split already over its share. For each split
    the target is ``desired_proportion * len(dataset_peptides)``; splits at or
    above target are *saturated* and get nothing, and the remaining new
    peptides are handed out in proportion to each unsaturated split's deficit.

    Args:
        dataset_peptides: Normalised peptides present in this dataset.
        existing_splits: Registry peptide sets; mutated in place with the new
            assignments.

    Returns:
        The updated split sets and the per-split count of peptides added.
    """
    total_dataset = len(dataset_peptides)
    if total_dataset == 0:
        return existing_splits, {split_name: 0 for split_name in _SPLITS}

    # Take the intersection of this dataset's peptides and the current registry peptides for each split
    existing_counts: Dict[str, int] = {
        split_name: len(dataset_peptides & existing_splits[split_name])
        for split_name in _SPLITS
    }

    saturated: Set[str] = set()
    deficits: Dict[str, float] = {}
    for split_name in _SPLITS:
        target = DESIRED_SPLIT_PROPORTIONS[split_name] * total_dataset
        if existing_counts[split_name] >= target:
            saturated.add(split_name)
            logger.warning(
                f"Split '{split_name}' is saturated: "
                f"{existing_counts[split_name]:,} existing >= "
                f"{target:,.0f} target "
                f"({DESIRED_SPLIT_PROPORTIONS[split_name]:.0%} "
                f"of {total_dataset:,})"
            )
        else:
            deficits[split_name] = target - existing_counts[split_name]

    all_known = set().union(*existing_splits.values())
    new_peptides = dataset_peptides - all_known
    if not new_peptides:
        return existing_splits, {split_name: 0 for split_name in _SPLITS}

    shuffled = sorted(new_peptides)
    rng = random.Random(SEED)
    rng.shuffle(shuffled)

    added: Dict[str, int] = {split_name: 0 for split_name in _SPLITS}
    total_deficit = sum(deficits.values())

    if total_deficit > 0:
        # Slice shuffled new peptides into contiguous chunks sized by each split's deficit
        offset = 0
        unsaturated = [s for s in _SPLITS if s not in saturated]
        for idx, split_name in enumerate(unsaturated):
            # Last split absorbs the remainder to avoid rounding drift
            if idx == len(unsaturated) - 1:
                count = len(shuffled) - offset
            else:
                count = round(len(shuffled) * deficits[split_name] / total_deficit)
            existing_splits[split_name].update(shuffled[offset : offset + count])
            added[split_name] = count
            offset += count
    else:
        # All splits saturated — fall back to train as a catch-all (this should never happen)
        existing_splits["train"].update(shuffled)
        added["train"] = len(shuffled)

    _log_dataset_peptide_proportions(dataset_peptides, existing_splits, total_dataset)
    return existing_splits, added


def _log_dataset_peptide_proportions(
    dataset_peptides: Set[str],
    existing_splits: Dict[str, Set[str]],
    total_dataset: int,
) -> None:
    """Log how far this dataset landed from the 80/10/10 target, given prior assignments."""
    logger.info(
        f"Split proportions for this dataset ({total_dataset:,} unique peptides):"
    )
    for split_name in _SPLITS:
        count = len(dataset_peptides & existing_splits[split_name])
        pct = count / total_dataset * 100 if total_dataset else 0
        target_pct = DESIRED_SPLIT_PROPORTIONS[split_name] * 100
        logger.info(
            f"  {split_name}: {count:,} ({pct:.1f}%)  [target: {target_pct:.1f}%]"
        )


# ── Split file writing ───────────────────────────────────────────────────────


def write_buffer(
    split: str,
    split_buffers: Dict[str, List[pl.DataFrame]],
    file_counters: Dict[str, int],
    buffer_sizes: Dict[str, int],
    output_dir: str,
) -> None:
    """Emit a shard once a split has buffered enough rows, shuffling away input-file order.

    Args:
        split: Split whose buffer should be flushed.
        split_buffers: Per-split accumulated frames; the flushed entry is reset.
        file_counters: Per-split shard counters, used for the output filename
            and incremented here.
        buffer_sizes: Per-split buffered row counts, reset for the flushed split.
        output_dir: Destination directory for the shard.
    """
    if not split_buffers[split]:
        return
    df = pl.concat(split_buffers[split], how="vertical_relaxed")
    df = df.sample(fraction=1.0, seed=SEED, shuffle=True)
    df = df.drop("normalised_peptide")
    out = os.path.join(output_dir, f"{split}_{file_counters[split]}.parquet")
    df.write_parquet(out)
    logger.info(f"Wrote {len(df):,} rows to {out}")
    split_buffers[split] = []
    buffer_sizes[split] = 0
    file_counters[split] += 1


def _flush_nonempty_split_buffers(
    split_buffers: Dict[str, List[pl.DataFrame]],
    file_counters: Dict[str, int],
    buffer_sizes: Dict[str, int],
    output_dir: str,
) -> None:
    """Make sure the final partial shard of each split still reaches disk."""
    for split_name in _SPLITS:
        if not split_buffers[split_name]:
            continue
        write_buffer(split_name, split_buffers, file_counters, buffer_sizes, output_dir)


def process_single_file(
    file_path: str,
    peptide_to_split: Dict[str, str],
    split_buffers: Dict[str, List[pl.DataFrame]],
    buffer_sizes: Dict[str, int],
    rows_per_file: int,
    output_dir: str,
    file_counters: Dict[str, int],
    column_rename: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    """Route one input file's rows to the split each peptide already belongs to.

    Every peptide must already be verified via :func:`verify_no_unseen_peptides`;
    an unregistered peptide raises a ``KeyError``.

    Args:
        file_path: Input parquet file to process.
        peptide_to_split: Normalised peptide to split-name lookup.
        split_buffers: Per-split accumulated frames, appended to in place.
        buffer_sizes: Per-split buffered row counts, updated in place.
        rows_per_file: Buffered rows that trigger a shard write.
        output_dir: Destination for any shard written during this call.
        file_counters: Per-split shard counters.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.

    Returns:
        Rows routed to buffers and the file's row count before filtering, so
        the caller can report how much was filtered out.
    """
    rename_map = _resolved_column_rename(column_rename)
    df = pl.read_parquet(file_path)
    df = _apply_schema_column_rename(df, rename_map)
    df = _ensure_unmodified_peptide(df)
    df = normalise_dataframe_schema(df, REFERENCE_SCHEMA, column_rename=None)
    row_count_before_filter = len(df)

    df = filter_spectra(df)
    if len(df) == 0:
        return 0, row_count_before_filter

    df = df.with_columns(
        pl.col("unmodified_peptide")
        .str.replace_all("I", "L")
        .alias("normalised_peptide")
    )

    labels = [peptide_to_split[p] for p in df["normalised_peptide"].to_list()]
    df = df.with_columns(pl.Series("_split", labels))

    output_row_count = 0
    for (split_name,), group in df.group_by("_split"):
        group = group.drop("_split")
        split_buffers[split_name].append(group)
        buffer_sizes[split_name] += len(group)
        output_row_count += len(group)
        if buffer_sizes[split_name] >= rows_per_file:
            write_buffer(
                split_name, split_buffers, file_counters, buffer_sizes, output_dir
            )

    return output_row_count, row_count_before_filter


def process_and_write_files(
    parquet_files: List[str],
    split_lookup: Dict[str, Set[str]],
    output_dir: str,
    rows_per_file: int,
    column_rename: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    """Stream the whole corpus into split shards without holding it in memory.

    Buffers rows per split and flushes as each reaches *rows_per_file*, so peak
    memory is bounded by the buffers rather than the corpus size.

    Args:
        parquet_files: Input files to process, in a fixed order.
        split_lookup: Registry peptide sets per split.
        output_dir: Destination for the split shards.
        rows_per_file: Buffered rows that trigger a shard write.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.

    Returns:
        Total spectra written and total spectra read, whose difference is what
        the quality filter removed.
    """
    rename_map = _resolved_column_rename(column_rename)
    peptide_to_split = _build_peptide_to_split(split_lookup)
    split_buffers, file_counters, buffer_sizes = _empty_split_buffer_state()

    start_time = time.time()
    total_files = len(parquet_files)
    total_output = 0
    total_input = 0

    for file_index, file_path in enumerate(parquet_files):
        _log_splitting_progress_if_due(file_index, total_files, start_time)

        output_row_count, input_row_count = process_single_file(
            file_path,
            peptide_to_split,
            split_buffers,
            buffer_sizes,
            rows_per_file,
            output_dir,
            file_counters,
            column_rename=rename_map,
        )
        total_output += output_row_count
        total_input += input_row_count

    _flush_nonempty_split_buffers(
        split_buffers, file_counters, buffer_sizes, output_dir
    )

    return total_output, total_input


# ── Verification ─────────────────────────────────────────────────────────────


def verify_no_unseen_peptides(
    parquet_files: List[str],
    split_lookup: Dict[str, Set[str]],
    dataset_peptides: Optional[Set[str]] = None,
    column_rename: Optional[Dict[str, str]] = None,
) -> None:
    """Fail before writing anything, rather than part-way through with unroutable rows.

    Args:
        parquet_files: Input files the split will cover.
        split_lookup: Registry peptide sets per split.
        dataset_peptides: Peptides already collected by a preceding pass; when
            omitted they are re-scanned from *parquet_files*.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.

    Raises:
        ValueError: If any dataset peptide is absent from the registry, listing
            a sample so the cause can be identified.
    """
    if dataset_peptides is None:
        dataset_peptides = collect_unique_peptides(
            parquet_files, column_rename=column_rename
        )

    all_known = set().union(*split_lookup.values())
    unseen = dataset_peptides - all_known

    if unseen:
        sample = sorted(unseen)[:10]
        raise ValueError(
            f"{len(unseen):,} peptide(s) not in registry. "
            f"Sample: {sample}{'...' if len(unseen) > 10 else ''}"
        )
    logger.info("Pre-split verification passed: all peptides are in the registry")


def verify_and_log(
    output_dir: str,
    total_input: int,
    split_lookup: Dict[str, Set[str]],
) -> None:
    """Show whether the run actually hit its split targets, and that no rows appeared from nowhere.

    Peptide proportions and spectral proportions differ (a split can hold 10%
    of peptides but a different share of spectra), so both are reported.

    Args:
        output_dir: Directory holding the written shards.
        total_input: Spectra read before quality filtering.
        split_lookup: Registry peptide sets per split.
    """
    total_peptides = sum(len(peptides) for peptides in split_lookup.values())
    if total_peptides:
        logger.info("Final peptide distribution:")
        for split_name in _SPLITS:
            count = len(split_lookup[split_name])
            pct = count / total_peptides * 100
            target_pct = DESIRED_SPLIT_PROPORTIONS[split_name] * 100
            logger.info(
                f"  {split_name}: {count:,} ({pct:.1f}%)"
                f"  [target: {target_pct:.1f}%]"
            )

    counts: Dict[str, int] = {}
    for split_name in _SPLITS:
        files = sorted(glob.glob(os.path.join(output_dir, f"{split_name}_*.parquet")))
        counts[split_name] = sum(
            pl.scan_parquet(f).select(pl.len()).collect().item() for f in files
        )

    total_out = sum(counts.values())
    logger.info("Final spectral distribution:")
    for split_name in _SPLITS:
        pct = counts[split_name] / total_out * 100 if total_out else 0
        logger.info(f"  {split_name}: {counts[split_name]:,} ({pct:.1f}%)")

    logger.info(f"Total input (pre-filter): {total_input:,}")
    logger.info(f"Total output:             {total_out:,}")
    logger.info(f"Filtered:                 {total_input - total_out:,}")

    if total_input >= total_out:
        logger.info("Verification passed")
    else:
        logger.warning(
            f"Output ({total_out:,}) exceeds input ({total_input:,}) — unexpected"
        )


# ── Orchestration ────────────────────────────────────────────────────────────


def run_update_splits(
    parquet_files: List[str],
    existing_splits: Dict[str, Set[str]],
    output_dir: str,
    upload_to_hf: bool,
    column_rename: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Register this dataset's peptides so the split pass has a decision for every row.

    Args:
        parquet_files: Input files to survey.
        existing_splits: Registry peptide sets per split, updated in place.
        output_dir: Where the registry parquet is written.
        upload_to_hf: Publish the updated registry to the shared dataset repo.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.

    Returns:
        The updated split sets and the dataset's peptide set, the latter so a
        following split pass can skip re-scanning the corpus.
    """
    logger.info("=== UPDATE-SPLITS ===")

    dataset_peptides = collect_unique_peptides(
        parquet_files, column_rename=column_rename
    )
    logger.info(f"Found {len(dataset_peptides):,} unique peptides")

    all_known = set().union(*existing_splits.values())
    new_peptides = dataset_peptides - all_known
    logger.info(f"New peptides: {len(new_peptides):,}")

    registry_changed = False
    if new_peptides:
        registry_changed = True
        existing_splits, added = assign_new_peptides(dataset_peptides, existing_splits)
        logger.info(
            f"Added {sum(added.values()):,}: "
            + ", ".join(f"{k}={v:,}" for k, v in added.items())
        )
    else:
        logger.info("No new peptides to add")

    os.makedirs(output_dir, exist_ok=True)
    save_registry(
        existing_splits,
        Path(output_dir) / REGISTRY_FILENAME,
        upload_to_hf,
        registry_changed=registry_changed,
    )
    return existing_splits, dataset_peptides


def run_split_only(
    parquet_files: List[str],
    existing_splits: Dict[str, Set[str]],
    output_dir: str,
    rows_per_file: int,
    dataset_peptides: Optional[Set[str]] = None,
    column_rename: Optional[Dict[str, str]] = None,
) -> None:
    """Write the split shards against an already-settled registry.

    Args:
        parquet_files: Input files to split.
        existing_splits: Registry peptide sets per split.
        output_dir: Destination for the split shards.
        rows_per_file: Buffered rows that trigger a shard write.
        dataset_peptides: Peptides from a preceding ``update-splits`` pass;
            supplying them lets verification skip a full re-scan.
        column_rename: Input-to-canonical column mapping; ``None`` uses the
            module default.
    """
    logger.info("=== SPLIT-ONLY ===")
    verify_no_unseen_peptides(
        parquet_files, existing_splits, dataset_peptides, column_rename=column_rename
    )

    os.makedirs(output_dir, exist_ok=True)
    total_output, total_input = process_and_write_files(
        parquet_files,
        existing_splits,
        output_dir,
        rows_per_file,
        column_rename=column_rename,
    )

    verify_and_log(output_dir, total_input, existing_splits)


def process_directories(
    input_dirs: List[str],
    output_dir: str,
    rows_per_file: int,
    registry_dir: Optional[str],
    mode: Mode,
    upload_to_hf: bool,
    column_remap_json: Optional[str] = None,
) -> None:
    """Run the requested mode over several directories as one dataset, so 80/10/10 is taken over the union.

    Args:
        input_dirs: Directories to walk for input parquet files.
        output_dir: Destination for the registry and/or split shards.
        rows_per_file: Buffered rows that trigger a shard write.
        registry_dir: Local registry directory; ``None`` downloads from
            HuggingFace.
        mode: Which of the registry update and split passes to run.
        upload_to_hf: Publish the updated registry to the shared dataset repo.
        column_remap_json: Raw ``--column-remap`` JSON, merged over the module
            default.
    """
    parquet_files = find_parquet_files(input_dirs)
    logger.info(
        f"Found {len(parquet_files):,} parquet files "
        f"across {len(input_dirs)} dir(s)"
    )
    if not parquet_files:
        logger.warning("No parquet files found")
        return

    column_rename = effective_column_rename(column_remap_json)

    _, existing_splits = load_peptide_registry(registry_dir)

    dataset_peptides: Optional[Set[str]] = None

    if mode in (Mode.UPDATE_SPLITS, Mode.BOTH):
        existing_splits, dataset_peptides = run_update_splits(
            parquet_files,
            existing_splits,
            output_dir,
            upload_to_hf,
            column_rename=column_rename,
        )

    if mode in (Mode.SPLIT_ONLY, Mode.BOTH):
        run_split_only(
            parquet_files,
            existing_splits,
            output_dir,
            rows_per_file,
            dataset_peptides=dataset_peptides,
            column_rename=column_rename,
        )

    logger.info("Processing complete")


# ── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def split(
    input_dir: str = INPUT_DIR_OPTION,
    output_dir: str = OUTPUT_DIR_OPTION,
    rows_per_file: int = ROWS_PER_FILE_OPTION,
    registry_dir: Optional[str] = REGISTRY_DIR_OPTION,
    mode: Mode = MODE_OPTION,
    upload_registry_to_hf: bool = UPLOAD_REGISTRY_OPTION,
    verbose: bool = VERBOSE_OPTION,
    column_remap: Optional[str] = COLUMN_REMAP_OPTION,
) -> None:
    """Split one labelled parquet directory into train/test/valid without peptide leakage.

    ``--mode split-only`` writes against the existing registry; ``update-splits`` updates the registry without writing shards.

    Args:
        input_dir: Directory of labelled parquet files to split.
        output_dir: Destination for the registry and split shards.
        rows_per_file: Rows per output shard.
        registry_dir: Local registry directory; omit to download from HuggingFace.
        mode: Whether to update the registry, write splits, or both.
        upload_registry_to_hf: Publish the updated registry to the shared repo.
        verbose: Raise logging to DEBUG for troubleshooting.
        column_remap: JSON mapping of source to canonical column names, for
            inputs using legacy names.

    Raises:
        typer.Exit: If the input directory is missing or ``--column-remap`` is
            not valid JSON, so neither failure surfaces mid-run.
    """
    configure_script_logging(verbose=verbose)
    if not os.path.exists(input_dir):
        typer.echo(f"Error: directory does not exist: {input_dir}")
        raise typer.Exit(1)

    try:
        parse_column_remap_json(column_remap)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    logger.info(f"Mode: {mode.value} | Input: {input_dir} | Output: {output_dir}")

    process_directories(
        input_dirs=[input_dir],
        output_dir=output_dir,
        rows_per_file=rows_per_file,
        registry_dir=registry_dir,
        mode=mode,
        upload_to_hf=upload_registry_to_hf,
        column_remap_json=column_remap,
    )


@app.command()
def batch(
    input_dirs: List[str] = INPUT_DIRS_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    rows_per_file: int = ROWS_PER_FILE_OPTION,
    registry_dir: Optional[str] = REGISTRY_DIR_OPTION,
    mode: Mode = MODE_OPTION,
    upload_registry_to_hf: bool = UPLOAD_REGISTRY_OPTION,
    verbose: bool = VERBOSE_OPTION,
    column_remap: Optional[str] = COLUMN_REMAP_OPTION,
) -> None:
    """Split several input directories as one dataset, in a single combined pass.

    Prefer this over repeated ``split`` runs when the directories belong to the
    same corpus: split proportions are then computed over their union, and each
    peptide is assigned once rather than one directory at a time. Non-existent
    directories are warned about and skipped.

    Args:
        input_dirs: Directories to treat as a single dataset.
        output_dir: Destination for the registry and split shards.
        rows_per_file: Rows per output shard.
        registry_dir: Local registry directory; omit to download from HuggingFace.
        mode: Whether to update the registry, write splits, or both.
        upload_registry_to_hf: Publish the updated registry to the shared repo.
        verbose: Raise logging to DEBUG for troubleshooting.
        column_remap: JSON mapping of source to canonical column names, for
            inputs using legacy names.

    Raises:
        typer.Exit: If no input directory exists or ``--column-remap`` is not
            valid JSON.
    """
    configure_script_logging(verbose=verbose)

    valid_dirs = [d for d in input_dirs if os.path.exists(d)]
    skipped = set(input_dirs) - set(valid_dirs)
    for d in skipped:
        logger.warning(f"Skipping non-existent directory: {d}")

    if not valid_dirs:
        typer.echo("Error: no valid input directories")
        raise typer.Exit(1)

    try:
        parse_column_remap_json(column_remap)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    logger.info(f"Mode: {mode.value} | Dirs: {valid_dirs} | Output: {output_dir}")

    process_directories(
        input_dirs=valid_dirs,
        output_dir=output_dir,
        rows_per_file=rows_per_file,
        registry_dir=registry_dir,
        mode=mode,
        upload_to_hf=upload_registry_to_hf,
        column_remap_json=column_remap,
    )


if __name__ == "__main__":
    app()
