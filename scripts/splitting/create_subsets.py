"""Build medium- and high-confidence parquet subsets from scored PSM tables."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator

import polars as pl
from tqdm import tqdm

from scripts.splitting.split_labelled_data import (
    normalise_dataframe_schema,
    REFERENCE_SCHEMA,
)

_TEMP_SCORING_COLS = ("_composite_score", "_peptide_length")


def get_reference_schema() -> dict[str, pl.DataType]:
    """Get the reference schema for the dataframe."""
    result: dict[str, pl.DataType] = REFERENCE_SCHEMA
    return result


def _drop_temp_scoring_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop the temporary scoring columns from the dataframe."""
    return df.drop(*_TEMP_SCORING_COLS)


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Find the column in the dataframe with the given candidates.

    Searches in the columns list for the given candidates and returns the first column that matches.
    Case-insensitive.g
    """
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        col = lower_to_original.get(candidate.lower())
        if col is not None:
            return col
    return None


def _pct_rank_avg_over(value: pl.Expr, by: pl.Expr) -> pl.Expr:
    """Min-max percentile rank within ``by``, mapped to [0, 1] with mean 0.5 for all group sizes.

    Uses ``(rank - 1) / (count - 1)`` so that the best item gets 1.0, the worst gets 0.0,
    and singletons (where ranking is meaningless) default to 0.5.  This avoids the upward
    bias of ``rank / count`` for small groups (e.g. singletons always scoring 1.0).
    """
    rank = value.rank(method="average").over(by)
    cnt = value.count().over(by)
    pct = pl.when(cnt <= 1).then(pl.lit(0.5)).otherwise((rank - 1) / (cnt - 1))
    return pct.fill_null(0.5)


def _peptide_length_expr(df: pl.DataFrame) -> pl.Expr:
    """Get the peptide length expression for the dataframe."""
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
    """Return ``df`` with ``_peptide_length`` and ``_composite_score`` columns added."""
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
    """True for rows to exclude: internal ``[IN:<int>]`` in sequence."""
    seq_name = _find_column(df.columns, ["sequence"])
    if seq_name is None:
        return pl.lit(False)
    s = pl.col(seq_name).cast(pl.Utf8, strict=False).fill_null("")
    return s.str.contains(r"\[[Ii][Nn]:\d+\]")


def _filter_out_glyco_sequences(df: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Filter out rows with glyco sequences if enabled."""
    if not enabled:
        return df
    return df.filter(~_sequence_contains_glyco_mods(df))


def filter_df(df: pl.DataFrame, score: float) -> pl.DataFrame:
    """Return rows whose composite score is strictly above ``score`` (temporary columns dropped)."""
    scored = with_composite_score(df)
    return _drop_temp_scoring_columns(scored.filter(pl.col("_composite_score") > score))


def iter_parquet_files(input_root: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(subfolder_name, file_name, path)`` for each parquet file under ``input_root``."""
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


def main() -> None:
    """Read all input parquets, set global score cutoffs, and write filtered outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Score PSM tables and write medium- (global top 10%%) and high-confidence "
            "(global top 2%%) parquet subsets, mirroring input subfolder layout."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing one subfolder per dataset with .parquet files",
    )
    parser.add_argument(
        "--medium-output-dir",
        required=True,
        help="Output root for the medium-confidence subset (e.g. former MCFM tree)",
    )
    parser.add_argument(
        "--high-output-dir",
        required=True,
        help="Output root for the high-confidence subset (e.g. former HCFM tree)",
    )
    parser.add_argument(
        "--hold-back-modified-rows",
        action="store_true",
        help=(
            "Exclude rows with internal modification tokens [IN:<int>] in sequence from "
            "scoring thresholds and outputs."
        ),
    )
    args = parser.parse_args()
    input_root = args.input_dir
    folder_mcfm = args.medium_output_dir
    folder_hcfm = args.high_output_dir
    hold_back = args.hold_back_modified_rows

    all_files = list(iter_parquet_files(input_root))
    if not all_files:
        print(f"No .parquet files found under: {input_root}")
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
        print("No valid composite scores were computed.")
        return

    all_scores_df = pl.concat(score_frames)
    threshold_mcfm = float(
        all_scores_df["_composite_score"].quantile(0.90, interpolation="linear")
    )
    threshold_hcfm = float(
        all_scores_df["_composite_score"].quantile(0.98, interpolation="linear")
    )
    print(f"Global MCFM threshold (top 10%): {threshold_mcfm:.6f}")
    print(f"Global HCFM threshold (top 2%): {threshold_hcfm:.6f}")

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

        print(f"Processing {subfolder}/{name}")
        print("Filter percentage mcfm: ", filtered_df_mcfm.height / df.height * 100)
        print("Filter percentage hcfm: ", filtered_df_hcfm.height / df.height * 100)

        filtered_df_mcfm.write_parquet(os.path.join(folder_mcfm, subfolder, name))
        filtered_df_hcfm.write_parquet(os.path.join(folder_hcfm, subfolder, name))

        count_input += df.height
        count_mcfm += filtered_df_mcfm.height
        count_hcfm += filtered_df_hcfm.height

    print(f"Total number of PSMs in input: {count_input}")
    print(f"Total number of PSMs in medium subset: {count_mcfm}")
    print(f"Total number of PSMs in high subset: {count_hcfm}")


if __name__ == "__main__":
    main()
