r"""Globally shuffle parquet splits that are far too large to hold in RAM.

Training needs rows in random order, but a corpus of sharded ``{split}_*.parquet``
files cannot simply be loaded and sampled. This script implements the two-pass
out-of-core shuffle from Jane Street's
https://blog.janestreet.com/how-to-shuffle-a-big-dataset/, which reaches a true
global shuffle while never holding more than a pile in memory.

Pass 1 — Scatter:
    For each input file, read it into memory, randomly assign every row to one
    of M piles, and write each pile to a temporary parquet file.  Sub-piles from
    different input files that share the same pile-id are then concatenated.

Pass 2 — Shuffle:
    Load each pile into RAM, shuffle it, and write it as a final output file.

Complexity (per split, e.g. the train split alone), with N = total rows across
the split's files and M = number of piles (≈ N / chunk_size):

    Time  : O(N)   — every row is read and written a constant number of times.
    RAM   : Pass 1 holds one input file at a time: O(max(rows per file)).
            Pass 2 loads one pile at a time (×2 for the shuffle copy): O(N/M).
    Disk  : Temporary piles use ≈ 1× the split's on-disk size.
            Output files also use ≈ 1× the split's on-disk size.
            Both coexist briefly, so peak transient disk ≈ 2× one split's size.
            Use ``--temp-dir`` to place pile files on a specific volume
            instead of the process default (often ``/tmp``).

Each split (train, valid, test) is processed independently and sequentially,
so these costs never stack across splits.

CLI::

    python scripts/splitting/shuffle_2pass.py --help

    # Forecast a safe --chunk-size for this machine before committing to a run
    python scripts/splitting/shuffle_2pass.py --forecast --sample-file train_0.parquet

    # Basic run: single directory
    python scripts/splitting/shuffle_2pass.py \
        --input-dir /data/shards --output-dir /data/shuffled

    # Combine multiple dataset directories into a single shuffled output
    python scripts/splitting/shuffle_2pass.py \
        --input-dir /data/dataset1 --input-dir /data/dataset2 \
        --input-dir /data/dataset3 --output-dir /data/shuffled

    # Custom chunk size + seed for reproducibility
    python scripts/splitting/shuffle_2pass.py \
        --input-dir /data/shards --output-dir /data/shuffled \
        --chunk-size 200000 --seed 42

    # Different parallelism per stage
    python scripts/splitting/shuffle_2pass.py \
        --input-dir /data/shards --output-dir /data/shuffled \
        --pass1-procs 4 --pass2-procs 8
"""

import glob
import hashlib
import logging
import multiprocessing as mp
from multiprocessing.pool import Pool as ProcessPool
import os
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, List, Optional, Union

import numpy as np
import polars as pl
import typer

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="2-pass out-of-core shuffle for large parquet splits",
    no_args_is_help=True,
    add_completion=False,
)
# ── Forecasting ──────────────────────────────────────────────────────────────


def estimate_bytes_per_row(sample_file: str, n: int = 10_000) -> float:
    """Ground the memory forecast in real data instead of the generic default.

    Args:
        sample_file: A representative parquet file from the corpus.
        n: Rows to read; enough to average out per-row size variation.

    Returns:
        Estimated in-memory bytes per row, falling back to 1000.0 if the file
        cannot be read, so forecasting never blocks a run.
    """
    try:
        df = pl.scan_parquet(sample_file).head(n).collect()
        bpr = float(df.estimated_size()) / len(df)
        logger.info(f"Estimated {bpr:.0f} bytes/row from {sample_file}")
        return bpr
    except Exception as e:
        logger.warning(f"Could not estimate bytes/row from {sample_file}: {e}")
        return 1000.0


def forecast_chunk_size(
    ram_gb: float,
    num_procs: int,
    bytes_per_row: float = 1000.0,
    safety: float = 0.7,
    pass2: bool = False,
) -> dict[str, Any]:
    """Size piles so a parallel run fits in RAM rather than OOMing hours in.

    Args:
        ram_gb: Total system RAM in GB.
        num_procs: Number of parallel workers competing for that RAM.
        bytes_per_row: Estimated in-memory bytes per row.
        safety: Fraction of RAM to use (0–1). Default 0.7 leaves 30% headroom.
        pass2: Account for 2× memory, since the shuffle holds the original and
            the shuffled copy at once.

    Returns:
        The recommended ``max_chunk_size`` plus a memory breakdown, so the
        forecast can be inspected before a long run is started.
    """
    usable = ram_gb * (1024**3) * safety
    per_proc = usable / num_procs
    overhead = 0.6  # leave room for Polars internals
    effective = per_proc * overhead
    multiplier = 2 if pass2 else 1
    chunk = int(effective / (bytes_per_row * multiplier))

    data_gb = (chunk * bytes_per_row * multiplier) / (1024**3)
    total_gb = data_gb / overhead * num_procs

    return {
        "max_chunk_size": chunk,
        "data_memory_per_process_gb": round(data_gb, 2),
        "total_memory_all_procs_gb": round(total_gb, 2),
        "ram_utilization_pct": round(total_gb / ram_gb * 100, 1),
        "bytes_per_row": bytes_per_row,
        "memory_multiplier": multiplier,
    }


def detect_ram_gb() -> float:
    """Discover the RAM budget so callers need not pass ``--ram-gb`` on every run.

    Returns:
        Total system RAM in GB.

    Raises:
        SystemExit: If *psutil* is not installed, since auto-detection is then
            impossible and the caller must supply ``--ram-gb`` explicitly.
    """
    try:
        import psutil

        gb = float(psutil.virtual_memory().total) / (1024**3)
        logger.info(f"Detected {gb:.1f} GB RAM")
        return gb
    except ImportError:
        raise SystemExit(
            "psutil is required for RAM auto-detection.  "
            "Install it (pip install psutil) or pass --ram-gb."
        )


# ── First pass helpers ───────────────────────────────────────────────────────


def _file_seed(seed: Optional[int], file_path: str) -> Optional[int]:
    """Derive a stable per-file seed from ``--seed`` and the path, so reruns match and parallel files do not share one RNG stream."""
    if seed is None:
        return None
    h = int(hashlib.sha256(file_path.encode()).hexdigest()[:8], 16)
    return seed + h


def _scatter_file(
    file_path: str, num_piles: int, temp_dir: str, seed: Optional[int]
) -> list[tuple[int, str, int]]:
    """Spread one file's rows across piles so pass 2 can shuffle globally in bounded RAM."""
    t0 = time.perf_counter()
    df = pl.read_parquet(file_path)
    n = len(df)

    rng = np.random.default_rng(_file_seed(seed, file_path))
    assignments = rng.integers(0, num_piles, size=n, dtype=np.int32)
    df = df.with_columns(pl.Series("__pile__", assignments))

    piles: list[tuple[int, str, int]] = []
    partitions = df.partition_by("__pile__", maintain_order=False, include_key=True)
    del df

    for part in partitions:
        pid = int(part["__pile__"][0])
        part = part.drop("__pile__")
        if len(part) == 0:
            continue
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        fname = f"pile_{path_hash}_{os.path.basename(file_path)}_{pid:04d}.parquet"
        path = os.path.join(temp_dir, fname)
        part.write_parquet(path)
        piles.append((pid, path, len(part)))

    dt = time.perf_counter() - t0
    logger.info(
        f"  Scattered {os.path.basename(file_path)}: "
        f"{n:,} rows -> {len(piles)} piles ({dt:.1f}s)"
    )
    return piles


def _combine_pile(
    pile_id: int, pile_files: list[tuple[str, int]], dest_dir: str
) -> Optional[tuple[int, str, int]]:
    """Gather a pile's contributions from every input file, moving rather than copying when there is only one."""
    total = sum(rc for _, rc in pile_files)
    if total == 0:
        return None

    out = os.path.join(dest_dir, f"final_pile_{pile_id:04d}.parquet")

    if len(pile_files) == 1:
        shutil.move(pile_files[0][0], out)
    else:
        dfs = [pl.scan_parquet(f) for f, _ in pile_files]
        pl.concat(dfs, how="vertical_relaxed").sink_parquet(out)
        for f, _ in pile_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return (pile_id, out, total)


# ── Second pass helpers ──────────────────────────────────────────────────────


def _shuffle_pile(
    pile_info: tuple[int, str, int],
    output_dir: str,
    split: str,
    seed: Optional[int],
) -> tuple[int, int]:
    """Randomise within a pile, which is what makes the scattered rows a true global shuffle."""
    pile_id, pile_file, _ = pile_info
    t0 = time.perf_counter()

    df = pl.read_parquet(pile_file)
    shuffle_seed = (seed + pile_id) if seed is not None else None
    shuffled = df.sample(
        n=len(df), with_replacement=False, seed=shuffle_seed, shuffle=True
    )
    del df

    out = os.path.join(output_dir, f"{split}_{pile_id}.parquet")
    shuffled.write_parquet(out)
    n = len(shuffled)
    dt = time.perf_counter() - t0
    logger.info(
        f"  Pile {pile_id}: {n:,} rows shuffled -> "
        f"{os.path.basename(out)} ({dt:.1f}s)"
    )
    return (pile_id, n)


# ── Orchestration ────────────────────────────────────────────────────────────


def _init_worker() -> None:
    """Restore logging in spawned workers, which start without the parent's configuration."""
    configure_script_logging(verbose=False)


def _spawn_pool(num_procs: int) -> ProcessPool:
    """Force the 'spawn' start method, so workers never inherit the parent's Polars thread state."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=num_procs, initializer=_init_worker)


def _count_rows(path: str) -> int:
    """Size the run from parquet metadata so planning does not read the whole corpus."""
    n: int = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return n


def _auto_chunk_size(
    bytes_per_row: float,
    pass2_procs: int,
    ram_gb: Optional[float] = None,
    safety: float = 0.7,
) -> int:
    """Pick a chunk size for the caller so an omitted ``--chunk-size`` cannot OOM pass 2."""
    if ram_gb is None:
        ram_gb = detect_ram_gb()
    info = forecast_chunk_size(ram_gb, pass2_procs, bytes_per_row, safety, pass2=True)
    chunk = max(1_000, int(info["max_chunk_size"]))
    logger.info(
        f"Auto chunk size: {chunk:,} rows "
        f"(RAM={ram_gb:.0f}GB, procs={pass2_procs}, "
        f"~{bytes_per_row:.0f}B/row, safety={safety:.0%})"
    )
    return chunk


def _run_parallel_or_seq(
    func: Callable[..., Any], items: list[Any], num_procs: int, label: str
) -> list[Any]:
    """Avoid paying process-spawn cost when there is nothing to parallelise."""
    if len(items) < 2:
        return [func(x) for x in items]
    n = min(num_procs, len(items))
    logger.info(f"  {label}: {len(items)} items × {n} procs")
    with _spawn_pool(n) as pool:
        return pool.map(func, items)


def shuffle_split(
    split_dirs: Union[str, list[str]],
    split: str,
    output_dir: str,
    chunk_size: Optional[int] = None,
    seed: Optional[int] = None,
    pass1_procs: Optional[int] = None,
    pass2_procs: Optional[int] = None,
    ram_gb: Optional[float] = None,
    bytes_per_row: float = 1000.0,
    safety: float = 0.7,
    temp_dir: Optional[str] = None,
) -> None:
    """Shuffle one split end to end when it does not fit in memory.

    Runs both passes, verifies the output row count against the input, and logs
    per-stage timings so a long run can be diagnosed after the fact.

    Args:
        split_dirs: One or more directories containing ``{split}_*.parquet``
            files.  When a list is given, files from every directory are combined
            into a single shuffled output.
        split: Split prefix to process, e.g. ``train``.
        output_dir: Where the shuffled ``{split}_{i}.parquet`` files are written.
        chunk_size: Target rows per pile; derived from RAM when omitted.
        seed: Fixes the scatter and shuffle randomness for reproducible output.
        pass1_procs: Worker count for the scatter pass; defaults to all CPUs.
        pass2_procs: Worker count for the shuffle pass; defaults to all CPUs.
        ram_gb: RAM budget used to derive *chunk_size*; auto-detected when omitted.
        bytes_per_row: Per-row size estimate feeding the same derivation.
        safety: Fraction of RAM the derivation is allowed to plan for.
        temp_dir: Parent directory for pass-1 temporary pile files (a unique
            subdirectory is created inside it). If ``None``, uses the system default
            (typically ``TMPDIR`` or ``/tmp``). Point it at a volume with room for
            roughly the split's on-disk size.
    """
    if isinstance(split_dirs, str):
        split_dirs = [split_dirs]
    files = sorted(
        f for d in split_dirs for f in glob.glob(os.path.join(d, f"{split}_*.parquet"))
    )
    if not files:
        logger.info(f"No {split}_*.parquet in {split_dirs} — skipping")
        return

    p1 = pass1_procs or mp.cpu_count()
    p2 = pass2_procs or mp.cpu_count()

    # ── Analyse input ────────────────────────────────────────────────────
    logger.info(f"[{split}] Analysing {len(files)} input files ...")
    file_rows = []
    for f in files:
        rc = _count_rows(f)
        file_rows.append(rc)
        logger.info(f"  {os.path.basename(f)}: {rc:,} rows")
    total = sum(file_rows)
    logger.info(f"  Total: {total:,} rows")

    # ── Chunk size & pile count ──────────────────────────────────────────
    if chunk_size is None:
        chunk_size = _auto_chunk_size(bytes_per_row, p2, ram_gb, safety)
    num_piles = max(1, total // chunk_size)
    logger.info(f"[{split}] {num_piles} piles (chunk ~ {chunk_size:,} rows)")

    os.makedirs(output_dir, exist_ok=True)
    if temp_dir is not None:
        os.makedirs(temp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        logger.info(f"[{split}] Temp pile directory: {tmp}")
        # ── Pass 1: scatter ──────────────────────────────────────────────
        t1 = time.perf_counter()
        logger.info(
            f"[{split}] Pass 1: scattering {len(files)} files "
            f"-> {num_piles} piles ..."
        )
        scatter = partial(_scatter_file, num_piles=num_piles, temp_dir=tmp, seed=seed)
        all_piles = _run_parallel_or_seq(scatter, files, p1, "Pass 1")
        dt1 = time.perf_counter() - t1
        logger.info(f"[{split}] Pass 1 done ({dt1:.1f}s)")

        # ── Combine sub-piles ────────────────────────────────────────────
        tc = time.perf_counter()
        buckets: dict[int, list[tuple[str, int]]] = {}
        for file_piles in all_piles:
            for pid, path, rc in file_piles:
                buckets.setdefault(pid, []).append((path, rc))

        combine_args = [(pid, buckets[pid], tmp) for pid in sorted(buckets)]
        if len(combine_args) < 2:
            final_piles = [
                r for a in combine_args if (r := _combine_pile(*a)) is not None
            ]
        else:
            n = min(p1, len(combine_args))
            with _spawn_pool(n) as pool:
                combined = pool.starmap(_combine_pile, combine_args)
            final_piles = [r for r in combined if r is not None]

        dtc = time.perf_counter() - tc
        logger.info(
            f"[{split}] Pile combine done ({dtc:.1f}s) — "  f"{len(final_piles)} piles"
        )

        # ── Pass 2: shuffle ──────────────────────────────────────────────
        t2 = time.perf_counter()
        logger.info(f"[{split}] Pass 2: shuffling {len(final_piles)} piles ...")
        shuf = partial(_shuffle_pile, output_dir=output_dir, split=split, seed=seed)
        results = _run_parallel_or_seq(shuf, final_piles, p2, "Pass 2")
        dt2 = time.perf_counter() - t2
        out_total = sum(r[1] for r in results)
        logger.info(f"[{split}] Pass 2 done ({dt2:.1f}s)")

    # ── Verify ───────────────────────────────────────────────────────────
    if out_total == total:
        logger.info(f"[{split}] Row counts match: {total:,}")
    else:
        logger.error(f"[{split}] Row mismatch: expected {total:,}, got {out_total:,}")

    wall = dt1 + dtc + dt2
    logger.info(
        f"[{split}] Wall time: "
        f"pass1={dt1:.1f}s  combine={dtc:.1f}s  pass2={dt2:.1f}s  "
        f"total={wall:.1f}s"
    )


def shuffle_all(
    base_dirs: Union[str, list[str]],
    output_dir: str,
    chunk_size: Optional[int] = None,
    seed: Optional[int] = None,
    pass1_procs: Optional[int] = None,
    pass2_procs: Optional[int] = None,
    ram_gb: Optional[float] = None,
    bytes_per_row: float = 1000.0,
    safety: float = 0.7,
    temp_dir: Optional[str] = None,
) -> None:
    """Shuffle a whole dataset directory in one call, split by split.

    Each directory in *base_dirs* must contain sharded parquet files:
    ``train_0.parquet``, ``train_1.parquet``, … and the same pattern for
    ``valid_``, ``val_``, and ``test_``.  When multiple directories are given,
    files from all of them are combined into a single shuffled output per split,
    which is how several source datasets get merged into one training corpus.
    A failure on one split is logged and the remaining splits still run.

    Args:
        base_dirs: One or more directories of sharded split files.
        output_dir: Destination for the shuffled shards of every split.
        chunk_size: Target rows per pile; derived from RAM when omitted.
        seed: Fixes randomness for reproducible output.
        pass1_procs: Worker count for the scatter pass; defaults to all CPUs.
        pass2_procs: Worker count for the shuffle pass; defaults to all CPUs.
        ram_gb: RAM budget used to derive *chunk_size*; auto-detected when omitted.
        bytes_per_row: Per-row size estimate feeding the same derivation.
        safety: Fraction of RAM the derivation is allowed to plan for.
        temp_dir: Parent directory for the temporary pile files.
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    for d in base_dirs:
        if not Path(d).is_dir():
            logger.error(f"Not a directory: {d}")
            return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"\n{'=' * 60}\n"  f"Shuffling shards in {', '.join(base_dirs)}\n"  f"{'=' * 60}"
    )
    dirs_as_str = [str(Path(d)) for d in base_dirs]
    for split in ("train", "valid", "val", "test"):
        try:
            shuffle_split(
                dirs_as_str,
                split,
                output_dir,
                chunk_size=chunk_size,
                seed=seed,
                pass1_procs=pass1_procs,
                pass2_procs=pass2_procs,
                ram_gb=ram_gb,
                bytes_per_row=bytes_per_row,
                safety=safety,
                temp_dir=temp_dir,
            )
        except Exception:
            logger.exception(f"Failed on {split} in {base_dirs}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _handle_forecast(
    ram_gb: Optional[float],
    bytes_per_row: float,
    sample_file: Optional[str],
    pass1_procs: Optional[int],
    pass2_procs: Optional[int],
    num_procs: Optional[int],
    safety: float,
) -> None:
    """Render the ``--forecast`` report so a chunk size can be chosen before committing hours."""
    ram = ram_gb or detect_ram_gb()
    bpr = bytes_per_row
    if sample_file and os.path.exists(sample_file):
        bpr = estimate_bytes_per_row(sample_file)

    p1 = pass1_procs or num_procs or mp.cpu_count()
    p2 = pass2_procs or num_procs or mp.cpu_count()

    lines = [
        f"\n{'=' * 60}",
        "CHUNK-SIZE FORECAST",
        f"{'=' * 60}",
        f"RAM: {ram:.1f} GB  |  bytes/row: {bpr:.0f}  |  safety: {safety:.0%}",
    ]

    for label, procs, is_p2 in [
        ("Pass 1 (scatter)", p1, False),
        ("Pass 2 (shuffle, 2× mem)", p2, True),
    ]:
        f = forecast_chunk_size(ram, procs, bpr, safety, pass2=is_p2)
        lines.append(f"\n{label}  ({procs} procs):")
        lines.append(f"  Max chunk size : {f['max_chunk_size']:>12,} rows")
        lines.append(f"  Mem / process  : {f['data_memory_per_process_gb']:>8.2f} GB")
        lines.append(f"  Mem total      : {f['total_memory_all_procs_gb']:>8.2f} GB")
        lines.append(f"  RAM utilisation : {f['ram_utilization_pct']:>7.1f}%")

    rec = min(
        forecast_chunk_size(ram, p1, bpr, safety, pass2=False)["max_chunk_size"],
        forecast_chunk_size(ram, p2, bpr, safety, pass2=True)["max_chunk_size"],
    )
    lines.append(f"\n{'=' * 60}")
    lines.append(f"Recommended --chunk-size: {rec:,}")
    lines.append(f"{'=' * 60}\n")
    logger.info("\n".join(lines))


@app.command()
def main(
    input_dir: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory with train_/valid_/test_ shards (repeatable)",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", help="Output directory for shuffled files"),
    ] = None,
    chunk_size: Annotated[
        Optional[int],
        typer.Option("--chunk-size", help="Target rows per pile (auto from RAM if omitted)"),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    num_procs: Annotated[
        Optional[int],
        typer.Option("--num-procs", help="Default process count for both passes"),
    ] = None,
    pass1_procs: Annotated[
        Optional[int],
        typer.Option("--pass1-procs", help="Processes for pass 1 (scatter)"),
    ] = None,
    pass2_procs: Annotated[
        Optional[int],
        typer.Option("--pass2-procs", help="Processes for pass 2 (shuffle)"),
    ] = None,
    ram_gb: Annotated[
        Optional[float],
        typer.Option("--ram-gb", help="Total system RAM in GB (auto-detected if omitted)"),
    ] = None,
    bytes_per_row: Annotated[
        float,
        typer.Option("--bytes-per-row", help="Estimated bytes per row"),
    ] = 1000.0,
    safety: Annotated[
        float,
        typer.Option("--safety", help="RAM safety factor 0–1"),
    ] = 0.7,
    sample_file: Annotated[
        Optional[Path],
        typer.Option("--sample-file", help="Parquet file to estimate bytes/row from"),
    ] = None,
    forecast: Annotated[
        bool,
        typer.Option("--forecast", help="Print chunk-size forecast and exit"),
    ] = False,
    temp_dir: Annotated[
        Optional[Path],
        typer.Option("--temp-dir", help="Parent directory for pass-1 temporary piles"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Shuffle sharded parquet splits that do not fit in RAM, using a 2-pass scatter/shuffle."""
    configure_script_logging(verbose=verbose)

    if forecast:
        _handle_forecast(
            ram_gb=ram_gb,
            bytes_per_row=bytes_per_row,
            sample_file=str(sample_file) if sample_file else None,
            pass1_procs=pass1_procs,
            pass2_procs=pass2_procs,
            num_procs=num_procs,
            safety=safety,
        )
        return

    dirs = [str(d) for d in (input_dir or [])]
    if not dirs or output_dir is None:
        raise typer.BadParameter(
            "--output-dir and at least one --input-dir are required (unless --forecast)"
        )

    shuffle_all(
        dirs,
        str(output_dir),
        chunk_size=chunk_size,
        seed=seed,
        pass1_procs=pass1_procs or num_procs,
        pass2_procs=pass2_procs or num_procs,
        ram_gb=ram_gb,
        bytes_per_row=bytes_per_row,
        safety=safety,
        temp_dir=str(temp_dir) if temp_dir else None,
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    app()
    logger.info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
