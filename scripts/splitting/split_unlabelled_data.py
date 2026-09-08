r"""Split unlabelled MS/MS spectra into train/val/test without leaking near-duplicates.

Unlabelled spectra have no peptide to key a split on, but the same analyte is
measured repeatedly, so a naive random split scatters near-identical spectra
across train and test and biases evaluation. Locality-Sensitive Hashing (LSH) gives
similar spectra the same hash, and each hash is assigned to exactly one split,
so an entire cluster of near-duplicates moves together. Assignments accumulate
in a persistent table (hundreds of millions of hashes), which later batches join
against so hashes seen before keep their original split.

Modes:
    full       — compute LSH hashes + split data (default)
    lsh_only   — compute and save LSH assignments only
    split_only — split using pre-computed LSH assignments

Checkpointing:
    Long-running jobs checkpoint every N files (default: 10).
    Resume by re-running the same command.  Use --clear-checkpoint to start fresh.

CLI::

    uv run python -m scripts.splitting.split_unlabelled_data --help
    uv run python -m scripts.splitting.split_unlabelled_data --mode full \
        --input-dir data --output-dir splits
    uv run python -m scripts.splitting.split_unlabelled_data --mode lsh_only \
        --input-dir data --output-dir splits
    uv run python -m scripts.splitting.split_unlabelled_data --mode split_only \
        --input-dir data --output-dir splits \
        --lsh-assignments splits/updated_lsh_assignments.parquet --mz-max 6000
"""

import logging
import glob
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import typer

from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection
from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Split unlabelled spectra using LSH clustering",
    no_args_is_help=True,
    add_completion=False,
)

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
LSH_BIN_STEP = 1.0
LSH_N_HYPERPLANES = 30
TARGET_LEN = 800

CHECKPOINT_FILENAME = "split_checkpoint.json"
PROGRESS_FILENAME = "split_progress.json"
CHUNK_SIZE = 200_000

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
    "experiment_name": pl.String,
}

_SCHEMA_COLS = list(REFERENCE_SCHEMA.keys())


# ── Checkpoint manager ───────────────────────────────────────────────────────


class CheckpointManager:
    """Lets a multi-day split survive interruption without redoing finished files.

    Persists the processed-file list alongside the in-flight split buffers, so a
    resumed run continues mid-shard rather than re-reading the corpus. A separate
    progress file is written every file for monitoring, since the checkpoint
    itself is only written every *interval* files.

    Args:
        output_dir: Directory the checkpoint, buffer pickle and progress file
            live in; created if absent.
        interval: Files between full checkpoints, trading rewind distance
            against the cost of pickling the buffers.
    """

    def __init__(self, output_dir: str, interval: int = 10):
        """Bind checkpoint paths so a resume finds the same files as the interrupted run."""
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.checkpoint_path = self.output_dir / CHECKPOINT_FILENAME
        self.progress_path = self.output_dir / PROGRESS_FILENAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        processed_files: List[str],
        split_buffers: Dict[str, List[pl.DataFrame]],
        buffer_sizes: Dict[str, int],
        file_counters: Dict[str, int],
        total_processed: int,
        total_input_spectra: int,
        start_time: float,
        mode: str,
        overall_mz_max: float,
        rows_per_file: int,
        parquet_files: List[str],
    ) -> None:
        """Capture enough state that a resumed run produces the same output as an uninterrupted one.

        Args:
            processed_files: Files already handled, skipped on resume.
            split_buffers: Rows buffered but not yet written, pickled so no
                spectra are lost mid-shard.
            buffer_sizes: Per-split buffered row counts.
            file_counters: Per-split shard counters, so resumed writes do not
                overwrite existing shards.
            total_processed: Spectra written so far.
            total_input_spectra: Spectra read so far.
            start_time: Original run start, kept so rate and ETA stay meaningful.
            mode: Run mode; a resume against a different mode is rejected.
            overall_mz_max: m/z ceiling the LSH projector was built with.
            rows_per_file: Shard size in effect.
            parquet_files: The full input list, compared on resume to detect a
                changed corpus.
        """
        meta = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "overall_mz_max": overall_mz_max,
            "rows_per_file": rows_per_file,
            "processed_files": processed_files,
            "buffer_sizes": buffer_sizes,
            "file_counters": file_counters,
            "total_processed": total_processed,
            "total_input_spectra": total_input_spectra,
            "start_time": start_time,
            "parquet_files": parquet_files,
            "checkpoint_version": "1.0",
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(self.output_dir / "split_buffers.pkl", "wb") as f:
            pickle.dump(split_buffers, f)
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Offer previous state to a restarted run, treating a damaged checkpoint as none.

        Returns:
            The saved state including any pickled ``split_buffers``, or ``None``
            when no checkpoint exists or it cannot be read — a corrupt
            checkpoint should start a fresh run, not abort one.
        """
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path) as f:
                data: Dict[str, Any] = json.load(f)
            buf_path = self.output_dir / "split_buffers.pkl"
            if buf_path.exists():
                with open(buf_path, "rb") as f:
                    data["split_buffers"] = pickle.load(f)
            logger.info(
                f"Checkpoint loaded — resuming from file "
                f"{len(data['processed_files'])}/{len(data['parquet_files'])}"
            )
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save_progress(
        self,
        current_file: str,
        file_index: int,
        total_files: int,
        total_processed: int,
        total_input_spectra: int,
        start_time: float,
    ) -> None:
        """Publish a per-file progress snapshot so a days-long run can be monitored.

        Args:
            current_file: File just finished.
            file_index: Position of that file in the input list.
            total_files: Size of the input list.
            total_processed: Spectra written so far.
            total_input_spectra: Spectra read so far.
            start_time: Run start, used to derive elapsed time and an estimate
                of the remaining time.
        """
        elapsed = time.time() - start_time
        with open(self.progress_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "current_file": current_file,
                    "file_index": file_index,
                    "total_files": total_files,
                    "progress_percent": (
                        (file_index / total_files) * 100 if total_files else 0
                    ),
                    "total_processed": total_processed,
                    "total_input_spectra": total_input_spectra,
                    "elapsed_time": elapsed,
                    "estimated_remaining": (
                        (total_files - file_index) / (file_index / elapsed)
                        if file_index > 0
                        else 0
                    ),
                },
                f,
                indent=2,
            )

    def clear_checkpoint(self) -> None:
        """Remove finished state so the next run does not resume into a completed job."""
        for p in (
            self.checkpoint_path,
            self.output_dir / "split_buffers.pkl",
            self.progress_path,
        ):
            if p.exists():
                p.unlink()
        logger.info("Checkpoint files cleared")


# ── Helpers ──────────────────────────────────────────────────────────────────


_LSH_ASSIGNMENT_FILENAMES = frozenset(
    {"lsh_assignments.parquet", "updated_lsh_assignments.parquet"}
)


def _is_spectrum_parquet(path: str) -> bool:
    """Stop the script's own LSH assignment tables being picked up as input spectra."""
    name = Path(path).name
    if name in _LSH_ASSIGNMENT_FILENAMES:
        return False
    parts = Path(path).parts
    return "lsh" not in {part.lower() for part in parts}


def find_files(input_dir: str) -> List[str]:
    """Collect the input corpus in a fixed order, which checkpoint resume depends on.

    Args:
        input_dir: Root to walk recursively.

    Returns:
        Sorted spectrum parquet paths, excluding LSH assignment tables that may
        sit in the same tree.

    Raises:
        ValueError: If *input_dir* does not exist.
    """
    path = Path(input_dir)
    if not path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    files = sorted(
        fp
        for fp in glob.glob(str(path / "**" / "*.parquet"), recursive=True)
        if _is_spectrum_parquet(fp)
    )
    logger.info(f"Found {len(files)} parquet files in {input_dir}")
    return files


def normalise_dataframe_schema(
    df: pl.DataFrame, schema: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """Give every source file the same columns, so their rows can share an output shard.

    Args:
        df: Frame to normalise.
        schema: Canonical column names and dtypes to conform to.

    Returns:
        The frame with exactly *schema*'s columns, in its order; missing ones
        null-filled and extras dropped.
    """
    missing = [
        pl.lit(None).cast(dtype).alias(name)
        for name, dtype in schema.items()
        if name not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)
    return df.select(list(schema.keys()))


def get_spectra(df: pl.DataFrame, target_len: int = TARGET_LEN) -> np.ndarray:
    """Turn variable-length peak lists into the fixed-shape array the LSH projector needs.

    Fills a pre-allocated array directly rather than going through Polars
    ``map_elements`` → per-row ``np.pad`` → ``np.stack``, which dominated
    runtime at corpus scale.

    Args:
        df: Frame carrying ``mz_array`` and ``intensity_array`` list columns.
        target_len: Peaks kept per spectrum; longer lists are truncated and
            shorter ones zero-padded.

    Returns:
        A ``(N, 2, target_len)`` float32 array of m/z and intensity channels.
    """
    n = len(df)
    out = np.zeros((n, 2, target_len), dtype=np.float32)
    for col_idx, col_name in enumerate(("mz_array", "intensity_array")):
        for i, arr in enumerate(df[col_name].to_list()):
            ln = min(len(arr), target_len)
            out[i, col_idx, :ln] = arr[:ln]
    return out


def filter_spectra(df: pl.DataFrame) -> pl.DataFrame:
    """Drop unusable spectra identically in both passes, so no row hashes to an unassigned split.

    Args:
        df: Spectra to filter.

    Returns:
        Rows within the retention time, offset, charge and m/z limits.
    """
    return df.filter(
        (pl.col("retention_time") <= 10800)
        & (pl.col("lower_offset") <= 300)
        & (pl.col("precursor_charge") >= 0)
        & (pl.col("precursor_charge") <= 7)
        & (pl.col("precursor_mz") <= 2000)
    )


def write_buffer(
    split: str,
    split_buffers: Dict[str, List[pl.DataFrame]],
    file_counters: Dict[str, int],
    buffer_sizes: Dict[str, int],
    output_dir: str,
) -> None:
    """Emit a shard once a split has buffered enough rows, shuffling away input-file order.

    Filenames are zero-padded to five digits so a lexical listing matches
    numeric order.

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
    # Zero-pad the counter (e.g. test_00001.parquet) so a lexical sort of the
    # output files matches numeric order. 5 digits covers >>9,999 files/split
    # (train already reaches ~3,400) and is fixed so widths never mix.
    out = Path(output_dir) / f"{split}_{file_counters[split]:05d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(out))
    split_buffers[split] = []
    file_counters[split] += 1
    buffer_sizes[split] = 0


def _iter_chunks(df: pl.DataFrame, chunk_size: int) -> Iterator[pl.DataFrame]:
    """Bound peak memory for large files without paying a disk round-trip."""
    if len(df) <= chunk_size:
        yield df
    else:
        yield from df.iter_slices(n_rows=chunk_size)


# ── Pass 2: process & distribute spectra ─────────────────────────────────────


def _process_chunk(
    df: pl.DataFrame,
    reference_schema: Dict[str, pl.DataType],
    split_buffers: Dict[str, List[pl.DataFrame]],
    buffer_sizes: Dict[str, int],
    rows_per_file: int,
    output_dir: str,
    file_counters: Dict[str, int],
    lsh_projector: BatchedPeakListRandomProjection,
    lsh_to_split: Dict,
) -> Tuple[int, int]:
    """Send each spectrum to the split its LSH cluster belongs to."""
    df = filter_spectra(df)
    if len(df) == 0:
        return 0, 0

    df = normalise_dataframe_schema(df, reference_schema)
    spectra = get_spectra(df)
    lsh_values = lsh_projector.compute(spectra, progress_bar=False)

    split_labels = [lsh_to_split[h] for h in lsh_values]
    df = df.with_columns(
        pl.Series("lsh", lsh_values),
        pl.Series("_split", split_labels),
    )

    total = 0
    for (split_name,), group in df.group_by("_split"):
        group = group.drop("_split")
        split_buffers[split_name].append(group)
        buffer_sizes[split_name] += len(group)
        total += len(group)
        if buffer_sizes[split_name] >= rows_per_file:
            write_buffer(
                split_name, split_buffers, file_counters, buffer_sizes, output_dir
            )

    n_unique = len(set(lsh_values.tolist()))
    return total, n_unique


def process_single_file(
    df: pl.DataFrame,
    reference_schema: Dict[str, pl.DataType],
    split_buffers: Dict[str, List[pl.DataFrame]],
    buffer_sizes: Dict[str, int],
    rows_per_file: int,
    output_dir: str,
    file_counters: Dict[str, int],
    lsh_projector: BatchedPeakListRandomProjection,
    lsh_to_split: Dict,
) -> Tuple[int, int]:
    """Route one input file's spectra to their splits, chunking so a large file cannot exhaust RAM.

    Args:
        df: The file's spectra, already read into memory.
        reference_schema: Canonical schema each chunk is normalised to.
        split_buffers: Per-split accumulated frames, appended to in place.
        buffer_sizes: Per-split buffered row counts, updated in place.
        rows_per_file: Buffered rows that trigger a shard write.
        output_dir: Destination for any shard written during this call.
        file_counters: Per-split shard counters.
        lsh_projector: Projector producing the cluster hash for each spectrum.
        lsh_to_split: Hash to split-name lookup covering every hash in this file.

    Returns:
        Spectra routed to buffers and the number of distinct hashes seen.
    """
    total_processed = 0
    total_unique = 0
    for chunk in _iter_chunks(df, CHUNK_SIZE):
        p, u = _process_chunk(
            chunk,
            reference_schema,
            split_buffers,
            buffer_sizes,
            rows_per_file,
            output_dir,
            file_counters,
            lsh_projector,
            lsh_to_split,
        )
        total_processed += p
        total_unique += u
    return total_processed, total_unique


# ── Pass 1: compute LSH hashes ──────────────────────────────────────────────


def _find_max_mz_value(parquet_files: List[str]) -> float:
    """Size the LSH binning grid to the data, capping at 6000 m/z so outliers cannot skew it."""
    logger.info("Finding maximum m/z value ...")
    max_vals: List[float] = []
    for i, f in enumerate(parquet_files):
        val = (
            pl.scan_parquet(f)
            .select(pl.col("mz_array").list.max().max())
            .collect()
            .item()
        )
        max_vals.append(val)
        if (i + 1) % 50 == 0 or i == len(parquet_files) - 1:
            logger.info(
                f"  Scanned {i + 1}/{len(parquet_files)} files, "
                f"current max m/z: {max(max_vals):.1f}"
            )

    overall: float = max(max_vals)
    logger.info(f"Maximum m/z value: {overall:.1f}")
    if overall > 6000:
        n_trunc = sum(v > 6000 for v in max_vals)
        logger.warning(
            f"Capping m/z at 6000 ({n_trunc}/{len(parquet_files)} files affected)"
        )
        overall = 6000.0
    return overall


def _process_file_for_lsh(
    df: pl.DataFrame,
    lsh_projector: BatchedPeakListRandomProjection,
) -> set:
    """Reduce a file to just its distinct cluster hashes."""
    unique: set = set()
    for chunk in _iter_chunks(df, CHUNK_SIZE):
        chunk = filter_spectra(chunk)
        if len(chunk) == 0:
            continue
        spectra = get_spectra(chunk)
        hashes = lsh_projector.compute(spectra, progress_bar=False)
        unique.update(hashes.tolist())
    return unique


# ── LSH assignment logic ────────────────────────────────────────────────────


_EMPTY_ASSIGNMENTS_SCHEMA: Dict[str, pl.DataType] = {
    "lsh_hash": pl.String,
    "split": pl.String,
}


def load_existing_lsh_assignments(
    existing_file: Optional[str], output_dir: str
) -> pl.DataFrame:
    """Read prior assignments so hashes seen in earlier batches keep the split they were given.

    Kept columnar rather than as a Python ``dict``: the historical table holds
    hundreds of millions of hashes and must never be materialised as Python
    objects.

    Args:
        existing_file: Path to the assignments parquet; defaults to
            ``lsh_assignments.parquet`` inside *output_dir*.
        output_dir: Run output directory, used to resolve the default path.

    Returns:
        A ``(lsh_hash, split)`` frame, empty but correctly typed when the file
        does not exist, so a first run needs no special casing.
    """
    if existing_file is None:
        existing_file = str(Path(output_dir) / "lsh_assignments.parquet")
    p = Path(existing_file)
    if not p.exists():
        logger.info(f"No existing LSH assignments at {p}")
        return pl.DataFrame(schema=_EMPTY_ASSIGNMENTS_SCHEMA)

    logger.info(f"Loading existing LSH assignments from {p}")
    df = pl.scan_parquet(str(p)).select(["lsh_hash", "split"]).collect()

    for row in df.group_by("split").len().iter_rows(named=True):
        logger.info(f"  {row['split']}: {row['len']:,} hashes")
    logger.info(f"  Total: {df.height:,}")
    return df


def _scan_existing_lsh_assignments(
    existing_file: Optional[str], output_dir: str, hash_dtype: pl.DataType
) -> pl.LazyFrame:
    """Keep historical LSH assignments lazy so hundred-million-row joins never collect."""
    if existing_file is None:
        existing_file = str(Path(output_dir) / "lsh_assignments.parquet")
    p = Path(existing_file)
    if not p.exists():
        logger.info(f"No existing LSH assignments at {p}")
        return pl.LazyFrame(schema={"lsh_hash": hash_dtype, "split": pl.String})

    # Row count comes from parquet metadata (cheap) rather than a full scan.
    total = pl.scan_parquet(str(p)).select(pl.len()).collect().item()
    logger.info(f"Loading existing LSH assignments from {p} ({total:,} hashes)")
    return (
        pl.scan_parquet(str(p))
        .select(["lsh_hash", "split"])
        .with_columns(pl.col("lsh_hash").cast(hash_dtype))
    )


def create_initial_lsh_assignments(
    all_lsh: pl.DataFrame, existing_assignments: pl.DataFrame
) -> pl.DataFrame:
    """Separate the batch's already-decided hashes from the genuinely new ones.

    A left join carries each known hash's split across and leaves brand-new
    hashes null — done columnar so it scales to 100M+ hashes.

    Args:
        all_lsh: One-column ``lsh_hash`` frame of the current batch's unique
            hashes.
        existing_assignments: Historical ``(lsh_hash, split)`` table.

    Returns:
        The batch's hashes with ``split`` filled where known and null where the
        hash has never been seen.
    """
    return all_lsh.select("lsh_hash").join(
        existing_assignments, on="lsh_hash", how="left"
    )


def assign_remaining_lsh(split_df: pl.DataFrame) -> pl.DataFrame:
    """Place new hashes so the batch approaches 80/10/10 despite immovable prior assignments.

    Already-assigned hashes cannot be moved without leaking, so a split can
    arrive over its target; new hashes are then handed out to fill the remaining
    deficits, scaled down proportionally when there are too few to go round.

    Args:
        split_df: The batch's hashes, with ``split`` null where undecided.

    Returns:
        The same hashes with every ``split`` filled in.
    """
    assigned_df = split_df.filter(pl.col("split").is_not_null())
    unassigned_df = split_df.filter(pl.col("split").is_null())

    total = len(assigned_df) + len(unassigned_df)
    target_train = round(total * TRAIN_RATIO)
    target_val = round(total * VAL_RATIO)
    target_test = total - target_train - target_val

    cur_train = len(assigned_df.filter(pl.col("split") == "train"))
    cur_val = len(assigned_df.filter(pl.col("split") == "val"))
    cur_test = len(assigned_df.filter(pl.col("split") == "test"))

    rem_train = max(0, target_train - cur_train)
    rem_val = max(0, target_val - cur_val)
    rem_test = max(0, target_test - cur_test)

    if any(
        c > t
        for c, t in [
            (cur_train, target_train),
            (cur_val, target_val),
            (cur_test, target_test),
        ]
    ):
        logger.warning(
            f"Some splits already exceed target: "
            f"train={cur_train}/{target_train}, val={cur_val}/{target_val}, "
            f"test={cur_test}/{target_test}"
        )

    logger.info(f"Remaining slots: train={rem_train}, val={rem_val}, test={rem_test}")

    shuffled = unassigned_df.sample(fraction=1.0, shuffle=True, seed=SEED)
    n_unassigned = len(shuffled)
    total_needed = rem_train + rem_val + rem_test

    if n_unassigned < total_needed:
        logger.warning(
            f"Need {total_needed} slots but only {n_unassigned} unassigned hashes"
        )
        scale = n_unassigned / total_needed if total_needed else 1
        rem_train = round(rem_train * scale)
        rem_val = round(rem_val * scale)
        rem_test = n_unassigned - rem_train - rem_val

    n_train = min(rem_train, n_unassigned)
    n_val = min(rem_val, n_unassigned - n_train)
    labels = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n_unassigned - n_train - n_val)
    )
    new_df = shuffled.with_columns(pl.Series("split", labels))
    return pl.concat([assigned_df, new_df])


def dict_to_dataframe(mapping: Dict) -> pl.DataFrame:
    """Lift a hash lookup back into the columnar form the assignment code works in.

    Args:
        mapping: ``{hash: split}`` lookup.

    Returns:
        The equivalent ``(lsh_hash, split)`` frame.
    """
    return pl.DataFrame(
        {"lsh_hash": list(mapping.keys()), "split": list(mapping.values())}
    )


def _updated_assignments_plan(
    existing_assignments: pl.DataFrame, split_df: pl.DataFrame
) -> pl.LazyFrame:
    """Merge history with this batch lazily so the combined table can stream to disk."""
    existing_only = existing_assignments.lazy().join(
        split_df.lazy().select("lsh_hash"), on="lsh_hash", how="anti"
    )
    return pl.concat([existing_only, split_df.lazy().select(["lsh_hash", "split"])])


def _log_split_distribution(split_df: pl.DataFrame, prefix: str) -> None:
    """Make the effect of an assignment pass visible, including how many hashes were new."""
    counts = {
        row["split"]: row["len"]
        for row in split_df.group_by("split").len().iter_rows(named=True)
    }
    logger.info(
        f"{prefix}: "
        f"train={counts.get('train', 0)}, "
        f"val={counts.get('val', 0)}, "
        f"test={counts.get('test', 0)}, "
        f"unassigned={counts.get(None, 0)}"
    )


def assign_batch_splits(
    all_lsh: pl.DataFrame, existing_assignments: pl.DataFrame
) -> pl.DataFrame:
    """Give every hash in the batch a split, which the splitting pass then relies on.

    Known hashes keep their existing split; brand-new hashes are assigned
    towards the ~80/10/10 ratio.

    Args:
        all_lsh: One-column ``lsh_hash`` frame of the batch's unique hashes.
        existing_assignments: Historical ``(lsh_hash, split)`` table.

    Returns:
        The ``(lsh_hash, split)`` table for the batch only.
    """
    logger.info("Creating LSH split assignments")
    split_df = create_initial_lsh_assignments(all_lsh, existing_assignments)
    _log_split_distribution(split_df, "Before assignment")
    split_df = assign_remaining_lsh(split_df)
    _log_split_distribution(split_df, "After assignment")
    return split_df


def create_and_verify_lsh_splits(
    all_lsh: pl.DataFrame, existing_assignments: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Assign the batch and produce the full updated table, materialised in memory.

    Suitable when the history is small enough to collect; production runs use
    :func:`_stream_lsh_assignments` instead.

    Args:
        all_lsh: One-column ``lsh_hash`` frame of the batch's unique hashes.
        existing_assignments: Historical ``(lsh_hash, split)`` table.

    Returns:
        The batch-only assignment table and the full historical-plus-new table.
    """
    split_df = assign_batch_splits(all_lsh, existing_assignments)
    updated = _updated_assignments_plan(existing_assignments, split_df).collect()
    return split_df, updated


def _stream_lsh_assignments(
    all_lsh_df: pl.DataFrame, existing_lf: pl.LazyFrame, out_path: Path
) -> pl.DataFrame:
    """Stream-merge batch hashes into history so 100M+ assignment rows never collect."""
    # Known hashes: existing rows whose hash is in this batch (build on the batch).
    known = existing_lf.join(all_lsh_df.lazy(), on="lsh_hash", how="semi").collect(
        engine="streaming"
    )
    # New hashes: batch hashes with no existing assignment.
    unknown = all_lsh_df.join(
        known.select("lsh_hash"), on="lsh_hash", how="anti"
    ).with_columns(pl.lit(None).cast(pl.String).alias("split"))
    split_df = pl.concat([known, unknown.select(["lsh_hash", "split"])])

    _log_split_distribution(split_df, "Before assignment")
    split_df = assign_remaining_lsh(split_df)
    _log_split_distribution(split_df, "After assignment")

    # Historical rows not in this batch keep their split; stream the union to disk.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_only = existing_lf.join(all_lsh_df.lazy(), on="lsh_hash", how="anti")
    # ``existing_lf`` may scan the very path we write to (the pipeline passes one
    # file for both --existing-lsh-assignments and --updated-lsh-assignments).
    # Writing in place truncates the file mid-scan — a hard panic locally and a
    # deadlock on a network mount — so sink to a sibling temp file and atomically
    # rename it into place once the scan is done.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    pl.concat(
        [existing_only, split_df.lazy().select(["lsh_hash", "split"])]
    ).sink_parquet(str(tmp_path))
    os.replace(tmp_path, out_path)
    return split_df


# ── Orchestration with checkpointing ─────────────────────────────────────────

_SPLITS = ("train", "val", "test")

_SplitOrchestrationState = Tuple[
    Dict[str, List[pl.DataFrame]],
    Dict[str, int],
    Dict[str, int],
    List[str],
    int,
    int,
    float,
]


def _fresh_state() -> _SplitOrchestrationState:
    """Provide the zeroed orchestration state a run starts from when there is no checkpoint."""
    return (
        {s: [] for s in _SPLITS},
        {s: 0 for s in _SPLITS},
        {s: 0 for s in _SPLITS},
        [],
        0,
        0,
        time.time(),
    )


def _init_split_state(
    ckpt: Optional[Dict[str, Any]],
    mode: str,
    mz_max: float,
    rows_per_file: int,
    parquet_files: List[str],
) -> _SplitOrchestrationState:
    """Only resume when the checkpoint was written under identical parameters."""
    if ckpt and ckpt.get("mode") == mode:
        if (
            ckpt["overall_mz_max"] == mz_max
            and ckpt["rows_per_file"] == rows_per_file
            and ckpt["parquet_files"] == parquet_files
        ):
            logger.info("Resuming from checkpoint ...")
            return (
                ckpt["split_buffers"],
                ckpt["buffer_sizes"],
                ckpt["file_counters"],
                ckpt["processed_files"],
                ckpt["total_processed"],
                ckpt["total_input_spectra"],
                ckpt["start_time"],
            )
        logger.warning("Checkpoint parameters mismatch — starting fresh")
    logger.info("Starting fresh processing ...")
    return _fresh_state()


def _calculate_split_totals(output_dir: str) -> Dict[str, int]:
    """Measure the written output from parquet metadata, so the final report costs no scan."""
    totals: Dict[str, int] = {}
    for split in _SPLITS:
        files = list(Path(output_dir).glob(f"{split}_*.parquet"))
        totals[split] = sum(
            pl.scan_parquet(str(f)).select(pl.len()).collect().item() for f in files
        )
    return totals


def _log_final_statistics(
    totals: Dict[str, int],
    total_processed: int,
    total_input: int,
    total_unique_lsh: int,
) -> None:
    """Warn when realised split ratios drift from target, since clustering can skew them."""
    total_rows = sum(totals.values())
    logger.info("Final split ratios:")
    for s in _SPLITS:
        ratio = totals[s] / total_rows if total_rows else 0
        target = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}[s]
        logger.info(f"  {s}: {totals[s]:,} ({ratio:.1%} vs target {target:.1%})")
        if abs(ratio - target) > 0.01:
            logger.warning(f"  {s} ratio {ratio:.1%} differs from target {target:.1%}")

    filtered = total_input - total_processed
    logger.info(
        f"Input: {total_input:,}  Filtered: {filtered:,}  Output: {total_processed:,}"
    )
    logger.info(
        f"LSH: {total_unique_lsh:,} unique hashes, "
        f"{total_processed / total_unique_lsh:.1f} spectra/hash"
    )


def process_and_write_files(
    parquet_files: List[str],
    output_dir: str,
    rows_per_file: int,
    overall_mz_max: float,
    lsh_to_split: Dict,
    checkpoint_manager: Optional[CheckpointManager] = None,
    mode: str = "split_only",
) -> Tuple[int, int]:
    """Write the split shards for the whole corpus, resumably.

    Checkpoints every *interval* files and once more on any failure, so an
    interrupted multi-day run restarts from the last good state instead of the
    beginning. The exception is re-raised after that checkpoint is written.

    Args:
        parquet_files: Input files to process, in a fixed order.
        output_dir: Destination for the split shards and checkpoint files.
        rows_per_file: Buffered rows that trigger a shard write.
        overall_mz_max: m/z ceiling the LSH projector is built with; must match
            the value used when the assignments were computed.
        lsh_to_split: Hash to split-name lookup covering every hash in the corpus.
        checkpoint_manager: Checkpoint handler; one is created for *output_dir*
            when omitted.
        mode: Mode label recorded in the checkpoint, so a resume can confirm it
            matches.

    Returns:
        Total spectra written and total spectra read.
    """
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(output_dir)

    ckpt = checkpoint_manager.load_checkpoint()
    (
        split_buffers,
        buffer_sizes,
        file_counters,
        processed_files,
        total_processed,
        total_input_spectra,
        start_time,
    ) = _init_split_state(ckpt, mode, overall_mz_max, rows_per_file, parquet_files)
    processed_set = set(processed_files)

    projector = BatchedPeakListRandomProjection(
        max_mz=np.ceil(overall_mz_max),
        bin_step=LSH_BIN_STEP,
        n_hyperplanes=LSH_N_HYPERPLANES,
        subbatch_size=1024,
        seed=SEED,
    )

    total_unique_lsh = len(lsh_to_split)

    for i, fp in enumerate(parquet_files):
        if fp in processed_set:
            continue

        logger.info(f"Processing file {i + 1}/{len(parquet_files)}: {fp}")
        try:
            df = pl.read_parquet(fp)
            total_input_spectra += len(df)

            processed, _ = process_single_file(
                df,
                REFERENCE_SCHEMA,
                split_buffers,
                buffer_sizes,
                rows_per_file,
                output_dir,
                file_counters,
                projector,
                lsh_to_split,
            )
            total_processed += processed
            processed_files.append(fp)
            processed_set.add(fp)

            checkpoint_manager.save_progress(
                fp,
                i + 1,
                len(parquet_files),
                total_processed,
                total_input_spectra,
                start_time,
            )
            if (i + 1) % checkpoint_manager.interval == 0:
                checkpoint_manager.save_checkpoint(
                    processed_files,
                    split_buffers,
                    buffer_sizes,
                    file_counters,
                    total_processed,
                    total_input_spectra,
                    start_time,
                    mode,
                    overall_mz_max,
                    rows_per_file,
                    parquet_files,
                )

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(f"  {processed:,} spectra ({rate:.0f} spectra/s cumulative)")

        except Exception as e:
            logger.error(f"Error processing {fp}: {e}")
            checkpoint_manager.save_checkpoint(
                processed_files,
                split_buffers,
                buffer_sizes,
                file_counters,
                total_processed,
                total_input_spectra,
                start_time,
                mode,
                overall_mz_max,
                rows_per_file,
                parquet_files,
            )
            raise

    for s in _SPLITS:
        if split_buffers[s]:
            write_buffer(s, split_buffers, file_counters, buffer_sizes, output_dir)

    _log_final_statistics(
        _calculate_split_totals(output_dir),
        total_processed,
        total_input_spectra,
        total_unique_lsh,
    )
    checkpoint_manager.clear_checkpoint()
    return total_processed, total_input_spectra


def _load_lsh_state_from_checkpoint_or_fresh(
    ckpt: Optional[Dict[str, Any]],
    output_dir: str,
) -> Tuple[List[str], Set[Any], int, float]:
    """Recover the accumulated hash set so a resumed LSH pass need not rehash finished files."""
    if ckpt and ckpt.get("mode") == "lsh_only":
        logger.info("Resuming LSH computation from checkpoint ...")
        processed_files: List[str] = ckpt["processed_files"]
        total_input_spectra: int = ckpt["total_input_spectra"]
        start_time: float = ckpt["start_time"]
        cache_path = Path(output_dir) / "lsh_cache.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                all_unique_lsh: Set[Any] = pickle.load(f)
            logger.info(f"Loaded {len(all_unique_lsh)} cached LSH values")
        else:
            all_unique_lsh = set()
        return processed_files, all_unique_lsh, total_input_spectra, start_time
    return [], set(), 0, time.time()


def _persist_lsh_only_interval_checkpoint(
    checkpoint_manager: CheckpointManager,
    output_dir: str,
    parquet_files: List[str],
    processed_files: List[str],
    total_input_spectra: int,
    start_time: float,
    overall_mz_max: float,
    all_unique_lsh: Set[Any],
) -> None:
    """Persist the hash set separately, since the LSH pass has no split buffers to save."""
    with open(checkpoint_manager.checkpoint_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "mode": "lsh_only",
                "overall_mz_max": overall_mz_max,
                "processed_files": processed_files,
                "total_input_spectra": total_input_spectra,
                "start_time": start_time,
                "parquet_files": parquet_files,
                "checkpoint_version": "1.0",
            },
            f,
            indent=2,
        )
    with open(Path(output_dir) / "lsh_cache.pkl", "wb") as f:
        pickle.dump(all_unique_lsh, f)
    logger.info(f"  LSH checkpoint: {len(all_unique_lsh):,} unique hashes")


def compute_lsh_assignments(
    parquet_files: List[str],
    output_dir: str,
    overall_mz_max: Optional[float] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    existing_lsh_assignments_path: Optional[str] = None,
    updated_lsh_assignments_path: Optional[str] = None,
) -> Tuple[pl.DataFrame, float]:
    """Decide which split every spectrum cluster belongs to, before any data is written.

    Hashes the whole corpus, merges the result into the historical assignment
    table, and streams the merged table to disk. Files that fail are recorded in
    ``failed_files.txt`` and skipped rather than aborting a multi-day pass.

    Args:
        parquet_files: Input files to hash, in a fixed order.
        output_dir: Destination for checkpoints, caches and the default
            assignment paths.
        overall_mz_max: m/z ceiling for the projector; scanned from the corpus
            when omitted.
        checkpoint_manager: Checkpoint handler; one is created for *output_dir*
            when omitted.
        existing_lsh_assignments_path: Historical assignments to inherit splits
            from; defaults to ``lsh_assignments.parquet`` in *output_dir*.
        updated_lsh_assignments_path: Where the merged table is streamed;
            defaults to ``updated_lsh_assignments.parquet`` in *output_dir*.

    Returns:
        The batch's ``(lsh_hash, split)`` table, used by the subsequent
        splitting pass, and the overall max m/z the projector was built with.
    """
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(output_dir)

    ckpt = checkpoint_manager.load_checkpoint()
    processed_files, all_unique_lsh, total_input_spectra, start_time = (
        _load_lsh_state_from_checkpoint_or_fresh(ckpt, output_dir)
    )

    processed_set = set(processed_files)

    if overall_mz_max is None:
        overall_mz_max = _find_max_mz_value(parquet_files)
    else:
        logger.info(f"Using provided max m/z: {overall_mz_max}")

    projector = BatchedPeakListRandomProjection(
        max_mz=np.ceil(overall_mz_max),
        bin_step=LSH_BIN_STEP,
        n_hyperplanes=LSH_N_HYPERPLANES,
        subbatch_size=1024,
        seed=SEED,
    )

    failed: List[str] = []
    for i, fp in enumerate(parquet_files):
        if fp in processed_set:
            continue

        logger.info(f"LSH {i + 1}/{len(parquet_files)}: {fp}")
        try:
            df = pl.read_parquet(fp)
            total_input_spectra += len(df)
            hashes = _process_file_for_lsh(df, projector)
            all_unique_lsh.update(hashes)
            processed_files.append(fp)
            processed_set.add(fp)

            checkpoint_manager.save_progress(
                fp,
                i + 1,
                len(parquet_files),
                len(all_unique_lsh),
                total_input_spectra,
                start_time,
            )
            if (i + 1) % checkpoint_manager.interval == 0:
                _persist_lsh_only_interval_checkpoint(
                    checkpoint_manager,
                    output_dir,
                    parquet_files,
                    processed_files,
                    total_input_spectra,
                    start_time,
                    overall_mz_max,
                    all_unique_lsh,
                )

        except Exception as e:
            logger.error(f"Failed on {fp}: {e}")
            failed.append(fp)

    if failed:
        fail_path = Path(output_dir) / "failed_files.txt"
        fail_path.write_text("\n".join(failed) + "\n")
        logger.warning(f"{len(failed)} files failed — see {fail_path}")

    if updated_lsh_assignments_path is None:
        updated_lsh_assignments_path = str(
            Path(output_dir) / "updated_lsh_assignments.parquet"
        )

    # Move the accumulated hashes out of the Python ``set`` into a columnar frame
    # and release the set. At production scale the set of SHA-256 hashes is many
    # GB; keeping it alive alongside the merge is what previously OOMed.
    n_unique_lsh = len(all_unique_lsh)
    all_lsh_df = pl.DataFrame({"lsh_hash": list(all_unique_lsh)})
    del all_unique_lsh

    # Existing assignments stay LAZY — the historical table has hundreds of
    # millions of rows (≈289M and growing) and must never be collected into
    # memory. ``lsh_hash`` is cast to the freshly-computed hash dtype (``S64``
    # byte strings → Binary) so the joins work even with no history yet.
    existing_lf = _scan_existing_lsh_assignments(
        existing_lsh_assignments_path, output_dir, all_lsh_df["lsh_hash"].dtype
    )

    # Merge + write with bounded memory (joins build only on the small batch side).
    out_path = Path(updated_lsh_assignments_path)
    split_df = _stream_lsh_assignments(all_lsh_df, existing_lf, out_path)
    del all_lsh_df
    logger.info(f"Saved LSH assignments to {out_path}")

    compression = n_unique_lsh / total_input_spectra if total_input_spectra else 0
    logger.info(
        f"LSH stats: {n_unique_lsh:,} unique hashes, "
        f"{total_input_spectra:,} input spectra, "
        f"compression {compression:.1%}"
    )

    checkpoint_manager.clear_checkpoint()
    cache = Path(output_dir) / "lsh_cache.pkl"
    if cache.exists():
        cache.unlink()

    return split_df, overall_mz_max


# ── CLI ──────────────────────────────────────────────────────────────────────


def _load_lsh_to_split(path: str) -> Dict:
    """Stream the hash-to-split lookup so a many-million-row assignments table does not OOM."""
    import pyarrow.parquet as pq

    split_intern: Dict[str, str] = {}
    lsh_to_split: Dict = {}
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        columns=["lsh_hash", "split"], batch_size=5_000_000
    ):
        hashes = batch.column("lsh_hash").to_pylist()
        splits = batch.column("split").to_pylist()
        for h, s in zip(hashes, splits):
            lsh_to_split[h] = split_intern.setdefault(s, s)
    logger.info(f"Loaded {len(lsh_to_split):,} LSH->split assignments")
    return lsh_to_split


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option("--input-dir", "-i", help="Directory of unlabelled parquet files"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Destination for LSH assignments and splits"),
    ] = Path("./splits"),
    mode: Annotated[
        str,
        typer.Option("--mode", help="lsh_only, split_only, or full"),
    ] = "full",
    rows_per_file: Annotated[
        int,
        typer.Option("--rows-per-file", help="Rows per output shard"),
    ] = 400_000,
    mz_max: Annotated[
        Optional[float],
        typer.Option("--mz-max", help="Max m/z (required for split_only)"),
    ] = None,
    lsh_assignments: Annotated[
        Optional[Path],
        typer.Option("--lsh-assignments", help="Precomputed LSH assignments parquet"),
    ] = None,
    existing_lsh_assignments: Annotated[
        Optional[Path],
        typer.Option(
            "--existing-lsh-assignments",
            help="Prior LSH table to extend when computing new hashes",
        ),
    ] = None,
    updated_lsh_assignments: Annotated[
        Optional[Path],
        typer.Option(
            "--updated-lsh-assignments",
            help="Where to write the updated LSH assignments table",
        ),
    ] = None,
    checkpoint_interval: Annotated[
        int,
        typer.Option("--checkpoint-interval", help="Save checkpoint every N files"),
    ] = 10,
    no_checkpoint: Annotated[
        bool,
        typer.Option("--no-checkpoint", help="Disable checkpointing"),
    ] = False,
    clear_checkpoint: Annotated[
        bool,
        typer.Option("--clear-checkpoint", help="Delete existing checkpoint and start fresh"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Split unlabelled spectra into train/val/test with near-duplicates kept in one split."""
    configure_script_logging(verbose=verbose)
    if mode not in ("lsh_only", "split_only", "full"):
        raise typer.BadParameter("--mode must be lsh_only, split_only, or full")

    _checkpoint = CheckpointManager(str(output_dir), interval=checkpoint_interval)
    if clear_checkpoint:
        _checkpoint.clear_checkpoint()
    ckpt_mgr: Optional[CheckpointManager] = None if no_checkpoint else _checkpoint

    parquet_files = find_files(str(input_dir))
    if not parquet_files:
        raise ValueError(f"No parquet files in {input_dir}")

    lsh_to_split: Optional[Dict] = None
    overall_mz_max: Optional[float] = None

    if mode in ("lsh_only", "full"):
        split_df, overall_mz_max = compute_lsh_assignments(
            parquet_files,
            str(output_dir),
            overall_mz_max=mz_max,
            checkpoint_manager=ckpt_mgr,
            existing_lsh_assignments_path=(
                str(existing_lsh_assignments) if existing_lsh_assignments else None
            ),
            updated_lsh_assignments_path=(
                str(updated_lsh_assignments) if updated_lsh_assignments else None
            ),
        )
        if mode == "lsh_only":
            logger.info("LSH assignment complete!")
            return
        lsh_to_split = dict(
            zip(split_df["lsh_hash"].to_list(), split_df["split"].to_list())
        )

    if mode in ("split_only", "full"):
        if mode == "split_only":
            if not lsh_assignments:
                raise ValueError("--lsh-assignments required for split_only")
            if mz_max is None:
                raise ValueError("--mz-max required for split_only")
            overall_mz_max = mz_max
            lsh_to_split = _load_lsh_to_split(str(lsh_assignments))

        assert lsh_to_split is not None
        assert overall_mz_max is not None

        process_and_write_files(
            parquet_files,
            str(output_dir),
            rows_per_file,
            overall_mz_max,
            lsh_to_split,
            checkpoint_manager=ckpt_mgr,
            mode="split_only",
        )
        logger.info("Splitting complete!")


if __name__ == "__main__":
    t0 = time.perf_counter()
    app()
    logger.info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
