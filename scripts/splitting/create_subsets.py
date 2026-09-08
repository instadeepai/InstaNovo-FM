"""Build medium- and high-confidence parquet subsets from scored PSM tables.

Search engines report several partly redundant confidence measures (expectation,
hyperscore, nextscore, probability), none of which is comparable across peptide
lengths. This script combines them into a single composite score that is
percentile-ranked within each peptide length, then applies two *global*
quantile cutoffs (top 10% and top 2%) across every input file so the resulting
subsets are consistent between datasets rather than per-file. The output mirrors
the input subfolder layout, so downstream splitting can be pointed at either
tree unchanged.

CLI::

    python scripts/splitting/create_subsets.py --help
    python scripts/splitting/create_subsets.py \
        --input-dir psms \
        --medium-output-dir subsets/medium \
        --high-output-dir subsets/high \
        --hold-back-modified-rows
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging
from scripts.splitting.split_labelled_data import (
    normalise_dataframe_schema,
    REFERENCE_SCHEMA,
)

logger = logging.getLogger(__name__)

_TEMP_SCORING_COLS = ("_composite_score", "_peptide_length")

app = typer.Typer(
    help="Build medium- and high-confidence parquet subsets from scored PSMs",
    no_args_is_help=True,
    add_completion=False,
)

def get_reference_schema() -> dict[str, pl.DataType]:
    """Get the canonical column schema so subsets stay compatible with the split pipeline.

    Returns:
        The same reference schema used by ``split_labelled_data``, suitable for
        passing to schema normalisation before writing a subset.
    """
    result: dict[str, pl.DataType] = REFERENCE_SCHEMA
    return result


def _drop_temp_scoring_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Keep scoring internals out of the written subsets."""
    return df.drop(*_TEMP_SCORING_COLS)


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Tolerate the case and naming variations search engines use for the same field."""
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        col = lower_to_original.get(candidate.lower())
        if col is not None:
            return col
    return None


def _pct_rank_avg_over(value: pl.Expr, by: pl.Expr) -> pl.Expr:
    """Rank within ``by`` as ``(rank - 1) / (count - 1)``, giving singletons 0.5 instead of the upward bias of ``rank / count``."""
    rank = value.rank(method="average").over(by)
    cnt = value.count().over(by)
    pct = pl.when(cnt <= 1).then(pl.lit(0.5)).otherwise((rank - 1) / (cnt - 1))
    return pct.fill_null(0.5)


def _peptide_length_expr(df: pl.DataFrame) -> pl.Expr:
    """Recover peptide length from whichever column carries it, so scores can be length-normalised."""
    cols = list(df.columns)
    length_name = _find_column(cols, ["peptide_length", "peptide length"])
    if length_name is not None:
        return pl.col(length_name).cast(pl.Float64, strict=False)
    peptide_name = _find_column(cols, ["peptide", "unmodified_peptide"])
    if peptide_name is not None:
        return (
            pl.col(peptide_name)
            .cast(pl.Utf8, strict=False)
            .str.count_matches(r"[A-Z]")
            .cast(pl.Float64)
        )
    return pl.lit(None).cast(pl.Float64)


def with_composite_score(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse the search engine's several confidence metrics into one comparable score.

    Each metric is percentile-ranked within its peptide-length group before
    averaging, so long and short peptides compete on equal terms. The
    hyperscore-minus-nextscore margin is folded in when ``nextscore`` is
    available.

    Args:
        df: PSM table carrying ``expectation``, ``probability`` and
            ``hyperscore`` (plus optionally ``nextscore``).

    Returns:
        The input frame with ``_peptide_length`` and ``_composite_score``
        added, ready for thresholding.
    """
    out = df.with_columns(_peptide_length_expr(df).alias("_peptide_length"))
    plen = pl.col("_peptide_length")
    expectation = pl.col("expectation").cast(pl.Float64, strict=False)
    probability = pl.col("probability").cast(pl.Float64, strict=False)
    hyperscore = pl.col("hyperscore").cast(pl.Float64, strict=False)
    flipped_exp = 1.0 - expectation
    n_exp = _pct_rank_avg_over(flipped_exp, plen)
    n_prob = _pct_rank_avg_over(probability, plen)
    n_hyp = _pct_rank_avg_over(hyperscore, plen)
    if "nextscore" in df.columns:
        nextscore = pl.col("nextscore").cast(pl.Float64, strict=False)
        delta = hyperscore - nextscore
        n_delta = _pct_rank_avg_over(delta, plen)
        composite = (n_exp + n_hyp + n_delta + n_prob) / 4.0
    else:
        composite = (n_exp + n_hyp + n_prob) / 3.0
    return out.with_columns(composite.alias("_composite_score"))


def _sequence_contains_glyco_mods(df: pl.DataFrame) -> pl.Expr:
    """Flag internal ``[IN:<int>]`` modification tokens, so those rows can be held back on request."""
    seq_name = _find_column(df.columns, ["sequence"])
    if seq_name is None:
        return pl.lit(False)
    s = pl.col(seq_name).cast(pl.Utf8, strict=False).fill_null("")
    return s.str.contains(r"\[[Ii][Nn]:\d+\]")


def _filter_out_glyco_sequences(df: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Optionally hold back glyco rows so they skew neither the thresholds nor the outputs."""
    if not enabled:
        return df
    return df.filter(~_sequence_contains_glyco_mods(df))


def filter_df(df: pl.DataFrame, score: float) -> pl.DataFrame:
    """Select the confident rows of a single table against a known score cutoff.

    Args:
        df: PSM table to score and filter.
        score: Composite-score cutoff; rows must score strictly above it.

    Returns:
        The surviving rows with the temporary scoring columns removed, so the
        result is writable as-is.
    """
    scored = with_composite_score(df)
    return _drop_temp_scoring_columns(scored.filter(pl.col("_composite_score") > score))


def iter_parquet_files(input_root: str) -> Iterator[tuple[str, str, str]]:
    """Walk the one-subfolder-per-dataset layout in a stable order so runs are reproducible.

    Args:
        input_root: Root directory holding one subfolder per dataset.

    Yields:
        ``(subfolder_name, file_name, path)`` triples, carrying the subfolder
        so the output tree can mirror the input layout.
    """
    for subfolder in sorted(os.listdir(input_root)):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for name in sorted(os.listdir(subfolder_path)):
            if not name.lower().endswith(".parquet"):
                continue
            file_path = os.path.join(subfolder_path, name)
            if os.path.isfile(file_path):
                yield subfolder, name, file_path


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Root directory with one subfolder per dataset of .parquet files",
        ),
    ],
    medium_output_dir: Annotated[
        Path,
        typer.Option(
            "--medium-output-dir",
            help="Output root for the medium-confidence subset (e.g. MCFM)",
        ),
    ],
    high_output_dir: Annotated[
        Path,
        typer.Option(
            "--high-output-dir",
            help="Output root for the high-confidence subset (e.g. HCFM)",
        ),
    ],
    hold_back_modified_rows: Annotated[
        bool,
        typer.Option(
            "--hold-back-modified-rows",
            help="Exclude rows with [IN:<int>] tokens from scoring and outputs",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Carve a PSM corpus into medium- and high-confidence subsets with corpus-wide cutoffs."""
    configure_script_logging(verbose=verbose)

    input_root = str(input_dir)
    folder_mcfm = str(medium_output_dir)
    folder_hcfm = str(high_output_dir)
    hold_back = hold_back_modified_rows

    all_files = list(iter_parquet_files(input_root))
    if not all_files:
        logger.info(f"No .parquet files found under: {input_root}")
        return

    # Pass 1: compute composite scores for all files to set global thresholds.
    score_frames: list[pl.DataFrame] = []
    for _subfolder, _name, file_path in tqdm(all_files, desc="Pass 1/2: scoring"):
        df = _filter_out_glyco_sequences(pl.read_parquet(file_path), hold_back)
        scored = with_composite_score(df)
        sf = scored.select(pl.col("_composite_score")).filter(
            pl.col("_composite_score").is_finite()
        )
        if sf.height > 0:
            score_frames.append(sf)

    if not score_frames:
        logger.info("No valid composite scores were computed.")
        return

    all_scores_df = pl.concat(score_frames)
    threshold_mcfm = float(
        all_scores_df["_composite_score"].quantile(0.90, interpolation="linear")
    )
    threshold_hcfm = float(
        all_scores_df["_composite_score"].quantile(0.98, interpolation="linear")
    )
    logger.info(f"Global MCFM threshold (top 10%): {threshold_mcfm:.6f}")
    logger.info(f"Global HCFM threshold (top 2%): {threshold_hcfm:.6f}")

    # Pass 2: apply global thresholds and write subsets.
    count_input = 0
    count_mcfm = 0
    count_hcfm = 0
    for subfolder, name, file_path in tqdm(all_files, desc="Pass 2/2: filtering"):
        os.makedirs(os.path.join(folder_mcfm, subfolder), exist_ok=True)
        os.makedirs(os.path.join(folder_hcfm, subfolder), exist_ok=True)

        df = pl.read_parquet(file_path)
        df = normalise_dataframe_schema(df, get_reference_schema())

        df = _filter_out_glyco_sequences(df, hold_back)
        scored = with_composite_score(df)
        filtered_df_mcfm = _drop_temp_scoring_columns(
            scored.filter(pl.col("_composite_score") > threshold_mcfm)
        )
        filtered_df_hcfm = _drop_temp_scoring_columns(
            scored.filter(pl.col("_composite_score") > threshold_hcfm)
        )

        logger.debug(f"Processing {subfolder}/{name}")
        logger.debug(f"Filter percentage mcfm: {filtered_df_mcfm.height / df.height * 100}")
        logger.debug(f"Filter percentage hcfm: {filtered_df_hcfm.height / df.height * 100}")

        filtered_df_mcfm.write_parquet(os.path.join(folder_mcfm, subfolder, name))
        filtered_df_hcfm.write_parquet(os.path.join(folder_hcfm, subfolder, name))

        count_input += df.height
        count_mcfm += filtered_df_mcfm.height
        count_hcfm += filtered_df_hcfm.height

    logger.info(f"Total number of PSMs in input: {count_input}")
    logger.info(f"Total number of PSMs in medium subset: {count_mcfm}")
    logger.info(f"Total number of PSMs in high subset: {count_hcfm}")


if __name__ == "__main__":
    app()
