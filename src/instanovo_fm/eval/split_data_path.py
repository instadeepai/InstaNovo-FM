"""Deterministically split one dataset into sequence-disjoint train/valid/test partitions."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional

import polars as pl
from sklearn.model_selection import train_test_split

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

DEFAULT_SEED = 42
DEFAULT_GROUP_KEY = "sequence"
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
_SPLIT_NAMES = ("train", "valid", "test")


def _split_dir(output_dir: str, split_name: str) -> str:
    """Return the per-split output subdirectory."""
    return os.path.join(output_dir, split_name)


def _split_glob(output_dir: str, split_name: str) -> str:
    """Return the parquet glob a downstream loader should read for ``split_name``."""
    return os.path.join(_split_dir(output_dir, split_name), "*.parquet")


def _existing_splits(output_dir: str) -> Optional[dict[str, str]]:
    """Return the split-path dict if all three split dirs already hold parquet files, else None."""
    paths = {}
    for split_name in _SPLIT_NAMES:
        if not glob.glob(_split_glob(output_dir, split_name)):
            return None
        paths[f"{split_name}_path"] = _split_glob(output_dir, split_name)
    return paths


def split_data_path(
    data_path: str,
    output_dir: str,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    group_key: str = DEFAULT_GROUP_KEY,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Split a single dataset into sequence-disjoint train/valid/test parquet directories.

    Args:
        data_path: Path (or glob / list-glob) to the single source dataset.
        output_dir: Directory under which ``train/``, ``valid/`` and ``test/`` subdirs are written.
        ratios: ``(train, valid, test)`` fractions of *unique sequences*. Must be positive and
            sum to 1.
        seed: Random seed for the reproducible sequence-level split.
        group_key: Metadata column defining the disjoint groups (default the peptide ``sequence``).
        max_samples: Optional cap on rows collected from the source before splitting (bounds memory).
        overwrite: If False (default) and all three split dirs already contain parquet files, reuse
            them without recomputing (the split is deterministic, so this is safe).

    Returns:
        Dict with keys ``train_path`` / ``valid_path`` / ``test_path``, each a ``*.parquet`` glob.

    Raises:
        ValueError: If ``ratios`` are invalid, if ``group_key`` is missing from the data, or if
            there are too few unique groups to form all three splits.
    """
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be positive and sum to 1, got {ratios}")

    if not overwrite:
        cached = _existing_splits(output_dir)
        if cached is not None:
            logger.info(f"Reusing existing splits in {output_dir} (pass overwrite=True to regenerate)")
            return cached

    logger.info(f"Loading dataset for splitting: {data_path}")
    load_source = S3FileHandler().download_parquet(data_path) if data_path.startswith("s3://") else data_path
    sdf = SpectrumDataFrame.load(load_source, lazy=True, is_annotated=True)
    df = sdf.collect_chunked(max_samples=max_samples, seed=seed)
    logger.info(f"Collected {len(df):,} rows")

    if group_key not in df.columns:
        raise ValueError(
            f"group_key '{group_key}' not found in dataset columns {df.columns}. Provide a sequence/group column present in the data via --group_key."
        )

    unique_groups = sorted(df[group_key].drop_nulls().unique().to_list())
    n_null = len(df) - df[group_key].drop_nulls().len()
    if n_null:
        logger.warning(f"{n_null:,} rows have a null '{group_key}' and are excluded from all splits")

    _, val_frac, test_frac = ratios
    if len(unique_groups) < len(_SPLIT_NAMES):
        raise ValueError(f"Need at least {len(_SPLIT_NAMES)} unique '{group_key}' values to split, got {len(unique_groups)}")

    train_groups, temp_groups = train_test_split(unique_groups, test_size=val_frac + test_frac, random_state=seed)
    val_groups, test_groups = train_test_split(temp_groups, test_size=test_frac / (val_frac + test_frac), random_state=seed)

    group_sets = {"train": set(train_groups), "valid": set(val_groups), "test": set(test_groups)}
    logger.info(
        f"Sequence-disjoint split of {len(unique_groups):,} unique '{group_key}': "
        f"train={len(train_groups):,} valid={len(val_groups):,} test={len(test_groups):,}"
    )

    result: dict[str, str] = {}
    for split_name in _SPLIT_NAMES:
        subset = df.filter(pl.col(group_key).is_in(list(group_sets[split_name])))
        target = _split_dir(output_dir, split_name)
        logger.info(f"Writing {split_name} split: {len(subset):,} rows to {target}")
        SpectrumDataFrame.from_polars(subset, is_annotated=True).save(target=Path(target), partition=split_name, name="split")
        result[f"{split_name}_path"] = _split_glob(output_dir, split_name)

    return result


def main() -> None:
    """Split a single dataset into sequence-disjoint train/valid/test parquet directories."""
    parser = argparse.ArgumentParser(description="Reproducibly split one dataset into sequence-disjoint train/valid/test parquet dirs")
    parser.add_argument("--data_path", required=True, help="Path (or glob) to the single source dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to write train/ valid/ test/ split subdirs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for the split (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--group_key",
        default=DEFAULT_GROUP_KEY,
        help=f"Column defining disjoint groups (default: {DEFAULT_GROUP_KEY})",
    )
    parser.add_argument("--train_frac", type=float, default=DEFAULT_RATIOS[0], help="Train fraction of unique groups")
    parser.add_argument("--val_frac", type=float, default=DEFAULT_RATIOS[1], help="Valid fraction of unique groups")
    parser.add_argument("--test_frac", type=float, default=DEFAULT_RATIOS[2], help="Test fraction of unique groups")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on rows collected before splitting (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate splits even if the output dirs already exist")
    args = parser.parse_args()

    paths = split_data_path(
        data_path=args.data_path,
        output_dir=args.output_dir,
        ratios=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        group_key=args.group_key,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )

    sys.stdout.write(f"{paths['train_path']}\t{paths['valid_path']}\t{paths['test_path']}\n")


if __name__ == "__main__":
    main()
