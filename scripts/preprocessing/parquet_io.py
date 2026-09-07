"""Shared parquet read/write helpers for preprocessing pipelines."""

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
    """Remove a trailing ``_0000-0017`` style shard suffix from a file stem."""
    match = SHARD_SUFFIX_RE.match(stem)
    if match:
        return match.group("base")
    return stem


def strip_mzml_suffix(stem: str) -> str:
    """Remove a trailing ``.mzml`` / ``.mzML`` embedded in a file stem."""
    return _MZML_SUFFIX_RE.sub("", stem)


def normalize_experiment_stem(stem: str) -> str:
    """Strip shard and embedded ``.mzml`` suffixes from a file stem."""
    return strip_mzml_suffix(strip_shard_suffix(stem))


def parquet_stem(filename: str) -> str:
    """Return the filename stem, without ``.parquet``, shard, or ``.mzml`` suffixes."""
    name = Path(filename).name
    if name.lower().endswith(".parquet"):
        name = name[: -len(".parquet")]
    return normalize_experiment_stem(name)


def merged_parquet_filename(filename: str) -> str:
    """Return the unsplit parquet filename for a whole file or shard member."""
    return f"{parquet_stem(filename)}.parquet"


def experiment_name_from_path(path_str: str) -> str:
    """Return the canonical experiment basename for parquet ``experiment_name`` / USI."""
    return parquet_stem(path_str)


_COMPOUND_SUFFIXES = (".mzml.ipc", ".mzml.gz", ".mzml.parquet")


def search_data_lookup_key(path_str: str) -> str:
    """Return a normalized key for matching Excel search-data ``file path`` entries."""
    from pathlib import PureWindowsPath

    name = PureWindowsPath(path_str).name
    for suffix in _COMPOUND_SUFFIXES:
        if name.lower().endswith(suffix):
            return normalize_experiment_stem(name[: -len(suffix)])
    return normalize_experiment_stem(Path(name).stem)


def parse_shard_suffix(filename: str) -> Optional[Tuple[int, int]]:
    """Return ``(shard_index, total_shards)`` from a ``_0000-0017`` suffix, or None."""
    stem = Path(filename).name
    if stem.lower().endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    match = SHARD_INDEX_RE.search(stem)
    if match is None:
        return None
    return int(match.group("shard")), int(match.group("total"))


def build_shard_path_map(parquet_paths: List[Path]) -> Dict[int, Path]:
    """Map shard index to local parquet path for one experiment."""
    mapping: Dict[int, Path] = {}
    for path in parquet_paths:
        parsed = parse_shard_suffix(path.name)
        shard_idx = 0 if parsed is None else parsed[0]
        mapping[shard_idx] = path
    return mapping


def build_shard_order(parquet_paths: List[Path]) -> List[Tuple[int, Path, int]]:
    """Return ``(shard_index, path, min_index)`` tuples sorted by shard index."""
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
    """Pick the shard whose existing ``index`` range contains *row_index*."""
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
    """Group shard filenames that belong to the same experiment."""
    groups: Dict[str, List[str]] = {}
    for filename in filenames:
        output_name = merged_parquet_filename(filename)
        groups.setdefault(output_name, [])
        if filename not in groups[output_name]:
            groups[output_name].append(filename)
    return {name: sorted(members) for name, members in sorted(groups.items())}


def nan_string_to_null_expr(column: str) -> pl.Expr:
    """Map string ``'nan'`` (any case) and float NaN to null; pass other values through."""
    return (
        pl.when(
            pl.col(column).cast(pl.String, strict=False).str.to_lowercase().eq("nan")
        )
        .then(None)
        .otherwise(pl.col(column))
    )


def atomic_write_parquet(df: pl.DataFrame, file_path: Path | str) -> None:
    """Write *df* to *file_path* atomically via a temporary file in the same directory."""
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
    """Replace string 'Unknown' with null in known metadata columns."""
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
    """Replace string ``'nan'`` with null in string columns cast to numeric dtypes."""
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
    """Cast *df* to *dtypes*, adding missing columns as null and dropping extras."""
    df = _replace_unknown_with_null(df)
    df = _replace_nan_strings_with_null(df, dtypes)
    exprs = []
    for name, dtype in dtypes.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(name))
    return df.select(exprs)
