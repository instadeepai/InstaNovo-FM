r"""Split unlabelled MS/MS spectra into train/val/test using LSH clustering.

Uses Locality-Sensitive Hashing to cluster similar spectra and assign entire
clusters to the same split, preventing data leakage.

Modes:
    full       — compute LSH hashes + split data (default)
    lsh_only   — compute and save LSH assignments only
    split_only — split using pre-computed LSH assignments

Usage:
    python split_unlabelled_data_v2.py --mode full --input-dir data --output-dir splits
    python split_unlabelled_data_v2.py --mode lsh_only --input-dir data --output-dir splits
    python split_unlabelled_data_v2.py --mode split_only --input-dir data --output-dir splits \
        --lsh-assignments splits/updated_lsh_assignments.parquet --mz-max 6000

Checkpointing:
    Long-running jobs checkpoint every N files (default: 10).
    Resume by re-running the same command.  Use --clear-checkpoint to start fresh.
"""

import argparse
import glob
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    """Manages checkpointing and resuming of the splitting process."""

    def __init__(self, output_dir: str, interval: int = 10):
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
        """Save current state to checkpoint file."""
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
        """Load checkpoint data if it exists."""
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
        """Save progress snapshot for monitoring."""
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
        """Clear checkpoint files after successful completion."""
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
    """Return False for LSH assignment tables stored alongside spectrum data."""
    name = Path(path).name
    if name in _LSH_ASSIGNMENT_FILENAMES:
        return False
    parts = Path(path).parts
    return "lsh" not in {part.lower() for part in parts}


def find_files(input_dir: str) -> List[str]:
    """Find all spectrum parquet files recursively under *input_dir*."""
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
    """Ensure *df* matches *schema* (add missing cols, reorder, drop extras)."""
    missing = [
        pl.lit(None).cast(dtype).alias(name)
        for name, dtype in schema.items()
        if name not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)
    return df.select(list(schema.keys()))


def get_spectra(df: pl.DataFrame, target_len: int = TARGET_LEN) -> np.ndarray:
    """Pad mz/intensity list columns into a (N, 2, target_len) float32 array.

    Directly fills a pre-allocated numpy array instead of going through
    Polars map_elements → per-row np.pad → np.stack.
    """
    n = len(df)
    out = np.zeros((n, 2, target_len), dtype=np.float32)
    for col_idx, col_name in enumerate(("mz_array", "intensity_array")):
        for i, arr in enumerate(df[col_name].to_list()):
            ln = min(len(arr), target_len)
            out[i, col_idx, :ln] = arr[:ln]
    return out


def filter_spectra(df: pl.DataFrame) -> pl.DataFrame:
    """Apply quality-filter criteria to spectra."""
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
    """Flush a split buffer to a shuffled parquet file."""
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
    """Yield in-memory chunks from *df* (no disk round-trip)."""
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
    """Process one (possibly chunked) DataFrame: filter → LSH → distribute."""
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
    """Process a file (auto-chunked in memory if > CHUNK_SIZE rows)."""
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
    """Find the maximum m/z value across all files (reads only mz_array col)."""
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
    """Compute LSH hashes for one file (auto-chunked in memory)."""
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
    """Load existing LSH assignments as a ``(lsh_hash, split)`` DataFrame.

    Returns an empty (but correctly-typed) DataFrame when no file exists. The
    assignments are kept columnar (not a Python ``dict``) so the historical
    table — which can hold hundreds of millions of hashes — never has to be
    materialised as Python objects.
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
    """Lazily scan existing LSH assignments as a ``(lsh_hash, split)`` LazyFrame.

    Unlike :func:`load_existing_lsh_assignments`, this never collects the table —
    the historical assignments hold hundreds of millions of rows, so callers join
    against it with the streaming engine (building only on the small batch side).
    ``lsh_hash`` is cast to *hash_dtype* to match the freshly-computed hashes, and
    an empty (typed) LazyFrame is returned when no file exists.
    """
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
    """Create assignment DataFrame: known hashes get their split, rest are null.

    *all_lsh* is a one-column ``lsh_hash`` DataFrame of the unique hashes seen in
    the current batch; *existing_assignments* is a ``(lsh_hash, split)`` table.
    A left join carries each known hash's split across and leaves brand-new
    hashes null — done columnar so it scales to 100M+ hashes.
    """
    return all_lsh.select("lsh_hash").join(
        existing_assignments, on="lsh_hash", how="left"
    )


def assign_remaining_lsh(split_df: pl.DataFrame) -> pl.DataFrame:
    """Assign unassigned hashes to achieve ~80/10/10 ratio."""
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
    """Convert {hash: split} dict to a two-column DataFrame."""
    return pl.DataFrame(
        {"lsh_hash": list(mapping.keys()), "split": list(mapping.values())}
    )


def _updated_assignments_plan(
    existing_assignments: pl.DataFrame, split_df: pl.DataFrame
) -> pl.LazyFrame:
    """Lazy plan for the full updated assignment table.

    Combines historical hashes that are *not* in the current batch (keeping
    their original split) with this batch's freshly-resolved assignments. The
    two sets are disjoint by construction, so no global de-duplication is
    needed. Kept lazy so callers can stream it straight to disk.
    """
    existing_only = existing_assignments.lazy().join(
        split_df.lazy().select("lsh_hash"), on="lsh_hash", how="anti"
    )
    return pl.concat([existing_only, split_df.lazy().select(["lsh_hash", "split"])])


def _log_split_distribution(split_df: pl.DataFrame, prefix: str) -> None:
    """Log the before/after train/val/test/unassigned counts for *split_df*."""
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
    """Resolve every hash in the current batch to a split.

    Known hashes keep their existing split; brand-new hashes are assigned to
    hit the ~80/10/10 ratio. Returns the ``(lsh_hash, split)`` table for the
    batch only.
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
    """Assign the current batch and merge it into the historical assignments.

    Returns ``(split_df, updated_assignments)`` where *split_df* covers only the
    current batch's hashes and *updated_assignments* is the full historical +
    new table. Both are columnar DataFrames.
    """
    split_df = assign_batch_splits(all_lsh, existing_assignments)
    updated = _updated_assignments_plan(existing_assignments, split_df).collect()
    return split_df, updated


def _stream_lsh_assignments(
    all_lsh_df: pl.DataFrame, existing_lf: pl.LazyFrame, out_path: Path
) -> pl.DataFrame:
    """Merge the current batch into the historical assignments with bounded memory.

    Every join builds its hashtable on the small current-batch frame and streams
    the huge historical table through it (semi/anti), and the final union is
    streamed straight to *out_path* — the 100M+-row historical table is never
    collected into memory. Returns the batch's ``(lsh_hash, split)`` table (needed
    by the downstream splitting pass).

    This is the memory-bounded production equivalent of
    :func:`create_and_verify_lsh_splits`.
    """
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
    """Resume from checkpoint or start fresh."""
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
    """Count rows per split from parquet metadata (no data read)."""
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
    """Process all files and write split outputs with checkpointing."""
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
    """Restore LSH-only run state from checkpoint, or start empty."""
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
    """Write JSON + pickle checkpoint for lsh_only mode."""
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
    """Compute LSH assignments for all spectra with checkpointing.

    Returns the ``(lsh_hash, split)`` table for the current batch (used by the
    subsequent splitting pass) and the overall max m/z. The full updated
    assignment table is streamed to *updated_lsh_assignments_path*.
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
    """Build the ``hash -> split`` lookup from an assignments parquet.

    The historical assignments table can hold 300M+ rows. Reading it whole with
    ``pl.read_parquet`` and then ``dict(zip(...))`` keeps the full DataFrame and
    a freshly-materialised dict alive at the same time (~90GB at 333M rows —
    OOM). Instead, stream the file in row-group batches so only one batch is held
    at a time, and share the handful of distinct split strings so that only the
    byte-string keys dominate memory (~40GB for 333M keys).
    """
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


def main() -> None:
    """Parse CLI arguments and run LSH assignment and/or splitting."""
    parser = argparse.ArgumentParser(
        description="Split unlabelled spectra using LSH clustering"
    )
    parser.add_argument(
        "--mode",
        choices=["lsh_only", "split_only", "full"],
        default="full",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="./splits")
    parser.add_argument("--rows-per-file", type=int, default=400_000)
    parser.add_argument("--mz-max", type=float, default=None)
    parser.add_argument("--lsh-assignments", default=None)
    parser.add_argument("--existing-lsh-assignments", default=None)
    parser.add_argument("--updated-lsh-assignments", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--clear-checkpoint", action="store_true")
    args = parser.parse_args()

    _checkpoint = CheckpointManager(args.output_dir, interval=args.checkpoint_interval)
    if args.clear_checkpoint:
        _checkpoint.clear_checkpoint()
    ckpt_mgr: Optional[CheckpointManager] = None if args.no_checkpoint else _checkpoint

    parquet_files = find_files(args.input_dir)
    if not parquet_files:
        raise ValueError(f"No parquet files in {args.input_dir}")

    lsh_to_split: Optional[Dict] = None
    overall_mz_max: Optional[float] = None

    if args.mode in ("lsh_only", "full"):
        split_df, overall_mz_max = compute_lsh_assignments(
            parquet_files,
            args.output_dir,
            overall_mz_max=args.mz_max,
            checkpoint_manager=ckpt_mgr,
            existing_lsh_assignments_path=args.existing_lsh_assignments,
            updated_lsh_assignments_path=args.updated_lsh_assignments,
        )
        if args.mode == "lsh_only":
            logger.info("LSH assignment complete!")
            return
        # ``full`` mode splits the data next, which needs a hash→split lookup.
        lsh_to_split = dict(
            zip(split_df["lsh_hash"].to_list(), split_df["split"].to_list())
        )

    if args.mode in ("split_only", "full"):
        if args.mode == "split_only":
            if not args.lsh_assignments:
                raise ValueError("--lsh-assignments required for split_only")
            if args.mz_max is None:
                raise ValueError("--mz-max required for split_only")
            overall_mz_max = args.mz_max
            lsh_to_split = _load_lsh_to_split(args.lsh_assignments)

        assert lsh_to_split is not None
        assert overall_mz_max is not None

        process_and_write_files(
            parquet_files,
            args.output_dir,
            args.rows_per_file,
            overall_mz_max,
            lsh_to_split,
            checkpoint_manager=ckpt_mgr,
            mode="split_only",
        )
        logger.info("Splitting complete!")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s")
