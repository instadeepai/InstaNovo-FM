r"""Split labelled MS/MS spectra into train/test/valid sets (optimised).

Partitions labelled mass spectrometry data while ensuring no peptide leakage.
Peptides are normalised (UNIMOD tags stripped, I→L) and looked up in a registry
loaded from HuggingFace or a local directory.

Modes:
    both (default) — update registry + split files
    update-splits  — update and save registry only
    split-only     — split files using existing registry

If *unmodified_peptide* is missing from an input file, it is filled from *sequence*
by removing bracketed segments (e.g. ``[UNIMOD:123]``) and all hyphen (``-``)
characters.

Usage:
    python split_labelled_data_v2.py split --input-dir lcfm --output-dir splits
    python split_labelled_data_v2.py split -i lcfm -o splits --mode split-only
    python split_labelled_data_v2.py split -i data -o splits \\
        --column-remap '{"legacy_peptide":"unmodified_peptide"}'
    python split_labelled_data_v2.py batch dir1 dir2 --output-dir splits/
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

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Quality filter config ────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityFilterConfig:
    """Quality filter configuration.

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
    """Mode of operation.

    Args:
        UPDATE_SPLITS: Update the splits and save the registry.
        SPLIT_ONLY: Split the data but do not update the registry.
        BOTH: Update the splits and save the registry.
    """

    UPDATE_SPLITS = "update-splits"
    SPLIT_ONLY = "split-only"
    BOTH = "both"


INPUT_DIR_OPTION = typer.Option("lcfm", "--input-dir", "-i")
OUTPUT_DIR_OPTION = typer.Option("lcfm_splits", "--output-dir", "-o")
ROWS_PER_FILE_OPTION = typer.Option(400_000, "--rows-per-file", "-r")
REGISTRY_DIR_OPTION = typer.Option(None, "--registry-dir")
MODE_OPTION = typer.Option(Mode.BOTH, "--mode", "-m")
UPLOAD_REGISTRY_OPTION = typer.Option(False, "--upload-registry-to-hf")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v")
INPUT_DIRS_ARG = typer.Argument(..., help="Input directories to process")
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
    """Parse *raw* as a JSON object of str→str; empty/whitespace → {}."""
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
    """Module defaults merged with CLI remap (CLI overrides duplicate keys)."""
    cli_part = parse_column_remap_json(cli_json)
    return {**REFERENCE_SCHEMA_COLUMN_REMAP, **cli_part}


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _resolved_column_rename(
    column_rename: Optional[Dict[str, str]],
) -> Dict[str, str]:
    return column_rename if column_rename is not None else REFERENCE_SCHEMA_COLUMN_REMAP


def _build_peptide_to_split(split_lookup: Dict[str, Set[str]]) -> Dict[str, str]:
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
    split_buffers: Dict[str, List[pl.DataFrame]] = {
        split_name: [] for split_name in _SPLITS
    }
    file_counters: Dict[str, int] = {split_name: 0 for split_name in _SPLITS}
    buffer_sizes: Dict[str, int] = {split_name: 0 for split_name in _SPLITS}
    return split_buffers, file_counters, buffer_sizes


def _log_splitting_progress_if_due(
    file_index: int, total_files: int, start_time: float
) -> None:
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
    """Find all parquet files recursively, sorted for reproducibility."""
    files: List[str] = []
    for d in input_dirs:
        for root, _, names in os.walk(d):
            for name in names:
                if name.endswith(".parquet"):
                    files.append(os.path.join(root, name))
    files.sort()
    return files


def _sequence_has_glyco_mods() -> pl.Expr:
    return (
        pl.col("sequence")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.contains(r"\[[Ii][Nn]:\d+\]")
    )


def _unmodified_peptide_from_sequence_expr() -> pl.Expr:
    """Strip bracketed modifications (square and round) and hyphens from *sequence*."""
    return (
        pl.col("sequence")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"\([^\)]*\)", "")
        .str.replace_all(r"\[[^\]]*\]", "")
        .str.replace_all("-", "")
    )


def _ensure_unmodified_peptide(df: pl.DataFrame) -> pl.DataFrame:
    """If *unmodified_peptide* is missing, create it from *sequence* (mods and hyphens stripped)."""
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
    """Lazy counterpart to :func:`_ensure_unmodified_peptide`."""
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
    """Rename columns toward canonical schema names.

    For each ``source -> target`` pair, renames *source* to *target* only when
    *source* is present and *target* is not (the existing target column wins).
    """
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
    """Lazy counterpart to :func:`_apply_schema_column_rename`."""
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
    """Build a filter expression that passes null values through unchanged.

    When a column was not present in the source data it is filled with nulls by
    schema normalisation.  Filtering on an all-null column should keep every row
    rather than dropping them all.
    """
    if op == "<=":
        cond = pl.col(col) <= threshold
    elif op == ">=":
        cond = pl.col(col) >= threshold
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return pl.col(col).is_null() | cond


def filter_spectra(df: pl.DataFrame) -> pl.DataFrame:
    """Apply quality-filter criteria to spectra.

    Null values in any filter column are treated as passing (not filtered out).
    This avoids dropping all rows when a column is absent from the source data
    and was filled with nulls during schema normalisation.
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
    """Add keys from *schema* missing on *df* as null columns with the given dtypes."""
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
    """Lazy counterpart to :func:`_add_missing_schema_columns`."""
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
    """Ensure *df* matches *schema* (optional rename, add missing cols, reorder).

    *column_rename* maps input column names to canonical *schema* names. If
    omitted, no renaming is performed (see :data:`REFERENCE_SCHEMA_COLUMN_REMAP`
    for the default map used by the split pipeline).

    When the caller has already applied renaming (and e.g. derived
    *unmodified_peptide*), pass ``column_rename=None`` so this step only fills
    missing columns and reorders.
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
    """Lazy counterpart to :func:`normalise_dataframe_schema`."""
    rename_map = column_rename if column_rename is not None else {}
    lf = _apply_schema_column_rename_lazy(lf, rename_map)
    lf = _add_missing_schema_columns_lazy(lf, schema)
    return lf.select(list(schema.keys()))


# ── Registry ─────────────────────────────────────────────────────────────────


def load_peptide_registry(
    registry_dir: Optional[str] = None,
) -> Tuple[pl.DataFrame, Dict[str, Set[str]]]:
    """Load peptide registry from HF or local dir.

    Returns (raw DataFrame, {train/test/valid: set of peptides}).
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
    """Save registry to parquet (uses 'validation' for HF schema compat).

    When *upload_to_hf* is true, the remote dataset is updated only if
    *registry_changed* is true (e.g. new peptides were assigned). If the
    registry matches what was already on Hugging Face, upload is skipped.
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
    """Collect unique normalised peptides (I→L) from files after quality filter."""
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
    """Assign peptides via deficit-proportional allocation with saturation.

    For each split, the target count is
    ``desired_proportion * len(dataset_peptides)``.  Splits already at or
    above their target are *saturated*; remaining new peptides are distributed
    proportionally to each unsaturated split's deficit.
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
    """Log per-split peptide proportions relative to this dataset."""
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
    """Flush a split buffer to a shuffled parquet file."""
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
    """Read, filter, normalise, and distribute rows to split buffers.

    All peptides must already be verified via ``verify_no_unseen_peptides``
    before calling this function.

    Returns (output_row_count, row_count_before_filter).
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
    """Process all files and write split outputs.

    Returns (total_output_spectra, total_input_spectra).
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
    """Verify every dataset peptide is in the registry before writing splits.

    Raises ``ValueError`` if any peptides are missing from the registry.
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
    """Report peptide and spectral distributions, verify row counts."""
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
    """Update the splits and save the registry.

    Returns (updated_splits, dataset_peptides).
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
    """Split data using the existing registry.

    If *dataset_peptides* is provided (e.g. from a preceding ``update-splits``
    pass), the pre-split verification reuses it instead of re-scanning.
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
    """Process the directories."""
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
    """Split parquet files into train/test/valid sets."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not os.path.exists(input_dir):
        typer.echo(f"Error: directory does not exist: {input_dir}")
        raise typer.Exit(1)

    try:
        parse_column_remap_json(column_remap)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Mode: {mode.value} | Input: {input_dir} | Output: {output_dir}")

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
    """Process multiple input directories in a single combined pass."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    valid_dirs = [d for d in input_dirs if os.path.exists(d)]
    skipped = set(input_dirs) - set(valid_dirs)
    for d in skipped:
        typer.echo(f"Warning: skipping non-existent directory: {d}")

    if not valid_dirs:
        typer.echo("Error: no valid input directories")
        raise typer.Exit(1)

    try:
        parse_column_remap_json(column_remap)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Mode: {mode.value} | Dirs: {valid_dirs} | Output: {output_dir}")

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
