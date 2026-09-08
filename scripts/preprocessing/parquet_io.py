"""Provide consistent Parquet naming, sharding, schema, and atomic-write helpers.

Preprocessing scripts import this module to avoid diverging conventions when
normalising experiment names, locating shards, or aligning table schemas. This
module is imported; it has no CLI.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

SHARD_SUFFIX_RE = re.compile(r"^(?P<base>.+)_\d{4}-\d{4}$")
SHARD_INDEX_RE = re.compile(r"_(?P<shard>\d{4})-(?P<total>\d{4})$")
_MZML_SUFFIX_RE = re.compile(r"\.mzml$", re.IGNORECASE)

_UNKNOWN_NULL_COLUMNS = ("collision_energy", "frag_type")


def strip_shard_suffix(stem: str) -> str:
    """Recover a shared experiment stem so shards can be grouped together.

    Args:
        stem: Filename stem that may end in a shard marker.

    Returns:
        Stem without a trailing shard marker.
    """
    match = SHARD_SUFFIX_RE.match(stem)
    if match:
        return match.group("base")
    return stem


def strip_mzml_suffix(stem: str) -> str:
    """Normalise embedded mzML suffixes before comparing experiment names.

    Args:
        stem: Filename stem that may retain an mzML suffix.

    Returns:
        Stem without the case-insensitive mzML suffix.
    """
    return _MZML_SUFFIX_RE.sub("", stem)


def normalize_experiment_stem(stem: str) -> str:
    """Produce a stable experiment identifier across source and shard filenames.

    Args:
        stem: Source or Parquet filename stem.

    Returns:
        Stem without shard or embedded mzML suffixes.
    """
    return strip_mzml_suffix(strip_shard_suffix(stem))


def parquet_stem(filename: str) -> str:
    """Derive the experiment key used to merge related Parquet shards.

    Args:
        filename: Whole-file or sharded Parquet filename.

    Returns:
        Canonical experiment stem.
    """
    name = Path(filename).name
    if name.lower().endswith(".parquet"):
        name = name[: -len(".parquet")]
    return normalize_experiment_stem(name)


def merged_parquet_filename(filename: str) -> str:
    """Give every shard the same destination name during dataset merging.

    Args:
        filename: Whole-file or sharded Parquet filename.

    Returns:
        Canonical unsplit Parquet filename.
    """
    return f"{parquet_stem(filename)}.parquet"


def experiment_name_from_path(path_str: str) -> str:
    """Keep Parquet experiment names and USI datafile components consistent.

    Args:
        path_str: Source or output data path.

    Returns:
        Canonical experiment basename.
    """
    return parquet_stem(path_str)


_COMPOUND_SUFFIXES = (".mzml.ipc", ".mzml.gz", ".mzml.parquet")


def search_data_lookup_key(path_str: str) -> str:
    """Match platform-specific data paths to Excel search-data entries.

    Args:
        path_str: POSIX or Windows-style path from data or metadata.

    Returns:
        Canonical filename key for metadata lookup.
    """
    from pathlib import PureWindowsPath

    name = PureWindowsPath(path_str).name
    for suffix in _COMPOUND_SUFFIXES:
        if name.lower().endswith(suffix):
            return normalize_experiment_stem(name[: -len(suffix)])
    return normalize_experiment_stem(Path(name).stem)


def parse_shard_suffix(filename: str) -> Optional[Tuple[int, int]]:
    """Expose shard numbering so callers can validate and order split datasets.

    Args:
        filename: Parquet filename that may include a shard marker.

    Returns:
        Shard index and declared total, or null for an unsplit file.
    """
    stem = Path(filename).name
    if stem.lower().endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    match = SHARD_INDEX_RE.search(stem)
    if match is None:
        return None
    return int(match.group("shard")), int(match.group("total"))


def build_shard_path_map(parquet_paths: List[Path]) -> Dict[int, Path]:
    """Enable direct lookup of the physical file containing a known shard.

    Args:
        parquet_paths: Paths belonging to one experiment.

    Returns:
        Shard indices mapped to local Parquet paths.
    """
    mapping: Dict[int, Path] = {}
    for path in parquet_paths:
        parsed = parse_shard_suffix(path.name)
        shard_idx = 0 if parsed is None else parsed[0]
        mapping[shard_idx] = path
    return mapping


def build_shard_order(parquet_paths: List[Path]) -> List[Tuple[int, Path, int]]:
    """Capture row-index boundaries needed to route updates back to shards.

    Args:
        parquet_paths: Paths belonging to one experiment.

    Returns:
        Shard index, path, and minimum row index tuples in shard order.
    """
    shards: List[Tuple[int, Path, int]] = []
    for path in parquet_paths:
        parsed = parse_shard_suffix(path.name)
        shard_idx = 0 if parsed is None else parsed[0]
        indices = pl.read_parquet(path, columns=["index"])["index"].to_list()
        min_index = min(int(value) for value in indices) if indices else 0
        shards.append((shard_idx, path, min_index))
    return sorted(shards, key=lambda item: item[0])


def shard_path_for_index(
    row_index: int, shard_order: List[Tuple[int, Path, int]]
) -> Path:
    """Route a row-level correction to the Parquet shard that owns it.

    Args:
        row_index: Existing dataset row index to locate.
        shard_order: Ordered shard boundaries from :func:`build_shard_order`.

    Returns:
        Path of the shard containing the row index.
    """
    if len(shard_order) == 1:
        return shard_order[0][1]

    for i, (_, path, _) in enumerate(shard_order):
        if i + 1 < len(shard_order):
            if row_index < shard_order[i + 1][2]:
                return path
        else:
            return path
    return shard_order[-1][1]


def group_parquet_filenames(filenames: List[str]) -> Dict[str, List[str]]:
    """Prepare deterministic experiment groups for merging split Parquet files.

    Args:
        filenames: Whole-file and sharded Parquet filenames.

    Returns:
        Canonical output names mapped to sorted member filenames.
    """
    groups: Dict[str, List[str]] = {}
    for filename in filenames:
        output_name = merged_parquet_filename(filename)
        groups.setdefault(output_name, [])
        if filename not in groups[output_name]:
            groups[output_name].append(filename)
    return {name: sorted(members) for name, members in sorted(groups.items())}


def nan_string_to_null_expr(column: str) -> pl.Expr:
    """Normalise textual and floating NaNs before strict schema conversion.

    Args:
        column: Column whose missing-value encodings should be unified.

    Returns:
        Polars expression that maps NaN representations to null.
    """
    return (
        pl.when(
            pl.col(column).cast(pl.String, strict=False).str.to_lowercase().eq("nan")
        )
        .then(None)
        .otherwise(pl.col(column))
    )


def atomic_write_parquet(df: pl.DataFrame, file_path: Path | str) -> None:
    """Avoid leaving a partial Parquet file when serialisation fails.

    Args:
        df: Materialised table to persist.
        file_path: Final Parquet destination.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _replace_unknown_with_null(df: pl.DataFrame) -> pl.DataFrame:
    """Turn ``Unknown`` in collision_energy/frag_type into null so those columns can stay nullable."""
    exprs = []
    for col in _UNKNOWN_NULL_COLUMNS:
        if col not in df.columns:
            continue
        if df.schema[col] == pl.String:
            exprs.append(
                pl.when(pl.col(col) == "Unknown")
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    if not exprs:
        return df
    return df.with_columns(exprs)


def _replace_nan_strings_with_null(
    df: pl.DataFrame, dtypes: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """Prevent textual NaNs from becoming invalid values during numeric casts."""
    exprs = []
    for name, dtype in dtypes.items():
        if name not in df.columns:
            continue
        if df.schema[name] == pl.String and dtype.is_numeric():
            exprs.append(nan_string_to_null_expr(name).alias(name))
    if not exprs:
        return df
    return df.with_columns(exprs)


def align_dataframe_to_schema(
    df: pl.DataFrame, dtypes: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """Make heterogeneous source tables safe to concatenate under one schema.

    Args:
        df: Source table to align.
        dtypes: Canonical column names and Polars data types.

    Returns:
        Table containing exactly the canonical columns and compatible types.
    """
    df = _replace_unknown_with_null(df)
    df = _replace_nan_strings_with_null(df, dtypes)
    exprs = []
    for name, dtype in dtypes.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(name))
    return df.select(exprs)
