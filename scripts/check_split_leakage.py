"""Quantify pretrain/finetune peptide leakage.

What fraction of *labelled* val/test peptides also appear in the ACFM
foundation-train set? The ACFM data carries peptide annotations
(``modified_peptide_unimod``) but is split by LSH (spectral clustering), so it
does not coordinate with the peptide-level labelled split — a peptide used to
evaluate the finetuned model may already be in the pretraining corpus.

Exact-spectrum matching is not viable here (labelled ``usi`` is null; ``scan``
differs in dtype and format between the two sides), so we compare peptides.
Peptides are normalised to unmodified, uppercased, with I->L collapsed
(isobaric), which is the standard de-novo leakage convention.

Reads only the peptide column and streams the large ACFM side so memory stays
bounded.

Usage (defaults target the ACFM layout on the mounts):
    python scripts/check_split_leakage.py
"""

from __future__ import annotations

import argparse
import glob
import logging
from typing import List, Optional

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("leakage")


def _files(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        out.extend(sorted(glob.glob(p, recursive=True)))
    return out


def _pick_peptide_col(cols: List[str], preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in cols:
            return c
    return None


def _norm_peptide(col: str) -> pl.Expr:
    """Modified/annotated peptide -> unmodified, uppercase, I->L.

    Strips ``[...]`` and ``(...)`` modification annotations (including any letters
    inside them) first, then removes any remaining non-alphabetic characters.
    """
    return (
        pl.col(col)
        .cast(pl.String)
        .str.replace_all(r"\[[^\]]*\]", "")
        .str.replace_all(r"\([^\)]*\)", "")
        .str.replace_all(r"[^A-Za-z]", "")
        .str.to_uppercase()
        .str.replace_all("I", "L")
        .alias("pep")
    )


def _peptides_lazy(files: List[str], col: str, per_file: bool) -> pl.LazyFrame:
    """Lazy frame of normalised, non-empty peptides from *files*.

    *per_file* concatenates per-file scans so heterogeneous schemas (the labelled
    sources differ in other columns' dtypes) don't trip a unified-scan collect.
    """
    if per_file:
        lf = pl.concat(
            [pl.scan_parquet(f).select(_norm_peptide(col)) for f in files],
            how="vertical",
        )
    else:
        lf = pl.scan_parquet(files).select(_norm_peptide(col))
    return lf.filter((pl.col("pep").is_not_null()) & (pl.col("pep") != ""))


def _describe(label: str, files: List[str], pep_cols: List[str]) -> List[str]:
    logger.info(f"[{label}] {len(files)} files")
    if not files:
        return []
    schema = pl.scan_parquet(files[0]).collect_schema()
    cols: List[str] = list(schema.names())
    logger.info(f"    columns: {cols}")
    col = _pick_peptide_col(cols, pep_cols)
    logger.info(f"    peptide column: {col}")
    if col is not None:
        sample = (
            pl.scan_parquet(files[0])
            .select(pl.col(col).cast(pl.String).alias("raw"), _norm_peptide(col))
            .head(3)
            .collect()
        )
        logger.info(f"    sample raw->norm: {sample.to_dicts()}")
    return cols


def main() -> None:
    """Report labelled val/test vs ACFM-train peptide overlap for the given globs."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--unlabelled-train",
        nargs="+",
        default=["fm_mount/acfm_splits/train_*.parquet"],
    )
    ap.add_argument(
        "--labelled",
        nargs="+",
        default=[
            "clean_acfm_mount/identity_splits_parquet/*_splits/val_*.parquet",
            "clean_acfm_mount/identity_splits_parquet/*_splits/test_*.parquet",
        ],
    )
    ap.add_argument(
        "--unlabelled-peptide-col",
        nargs="+",
        default=["modified_peptide_unimod", "sequence", "unmodified_peptide"],
    )
    ap.add_argument(
        "--labelled-peptide-col",
        nargs="+",
        default=["unmodified_peptide", "sequence", "modified_peptide_unimod"],
    )
    args = ap.parse_args()

    logger.info("=== DISCOVERY ===")
    us_files = _files(args.unlabelled_train)
    lab_files = _files(args.labelled)
    us_cols = _describe("ACFM train", us_files, args.unlabelled_peptide_col)
    lab_cols = _describe("labelled val/test", lab_files, args.labelled_peptide_col)
    if not us_files or not lab_files:
        logger.warning("One side is empty — check the paths above.")
        return

    us_col = _pick_peptide_col(us_cols, args.unlabelled_peptide_col)
    lab_col = _pick_peptide_col(lab_cols, args.labelled_peptide_col)
    if us_col is None or lab_col is None:
        logger.warning(f"No peptide column (ACFM={us_col}, labelled={lab_col}).")
        return

    logger.info(f"=== PEPTIDE OVERLAP (I=L)  ACFM.{us_col} vs labelled.{lab_col} ===")

    # Small side: unique normalised peptides in labelled val/test.
    lab_peps = (
        _peptides_lazy(lab_files, lab_col, per_file=True)
        .unique()
        .collect(engine="streaming")
    )
    n_lab = lab_peps.height
    logger.info(f"Labelled val/test unique peptides (I=L): {n_lab:,}")

    # Stream ACFM train, keep rows whose peptide is in the labelled set (build the
    # hashtable on the small labelled side), then count distinct leaked peptides.
    leaked = (
        _peptides_lazy(us_files, us_col, per_file=False)
        .join(lab_peps.lazy(), on="pep", how="semi")
        .unique()
        .collect(engine="streaming")
    )
    n_leaked = leaked.height
    pct = (n_leaked / n_lab * 100) if n_lab else 0.0
    logger.info("")
    logger.info(
        f"PEPTIDE LEAKAGE: {n_leaked:,} / {n_lab:,} labelled val/test peptides "
        f"({pct:.2f}%) also appear in the ACFM pretraining-train set"
    )
    logger.info(f"Example leaked peptides: {leaked['pep'].head(10).to_list()}")


if __name__ == "__main__":
    main()
