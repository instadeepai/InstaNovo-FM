"""Set ``frag_type`` from the activation methods in each spectrum's scan header.

Conversion took ``frag_type`` from the first of a spectrum's activation methods. Spectra activated twice
therefore lost one method: EThcD (``@etd..@hcd..`` in the header) and ETciD (``@etd..@cid..``) were stored as
``ETD``, ``HCD`` or ``CID``. The Thermo filter string in ``header`` lists every method, so this script
derives ``frag_type`` from it:

========================  ==========
header activations        frag_type
========================  ==========
``@hcd``                  ``HCD`` (an existing ``HCID`` is kept: OpenMS's name for the same method)
``@cid``                  ``CID``
``@etd``                  ``ETD``
``@etd`` + ``@hcd``       ``EThcD``
``@etd`` + ``@cid``       ``ETciD``
anything else             the methods upper-cased and joined with ``+``, e.g. ``UVPD``
========================  ==========

Rows without a header, or whose header names no activation, keep their ``frag_type``. The mass analyser
(``FTMS``/``ITMS`` at the start of the header) is not part of ``frag_type``.

CLI::

    uv run python -m scripts.preprocessing.frag_type_from_header --help
    uv run python -m scripts.preprocessing.frag_type_from_header --input-dir <data-root>/lcfm --dry-run
    uv run python -m scripts.preprocessing.frag_type_from_header --input-dir <data-root>/lcfm --output-file frag_type_changes.csv

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Annotated, List, Optional

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Set frag_type from the activation methods in the scan header",
    no_args_is_help=True,
    add_completion=False,
)

COMBINED = {"etd+hcd": "EThcD", "etd+cid": "ETciD"}


def header_frag_type(header: pl.Expr) -> pl.Expr:
    """The fragmentation type a Thermo filter string names, or null when it names no activation method."""
    methods = header.str.extract_all(r"@[a-z]+").list.eval(pl.element().str.slice(1)).list.join("+")
    return (
        pl.when(methods.is_null() | (methods == ""))
        .then(pl.lit(None, dtype=pl.String))
        .when(methods.is_in(list(COMBINED)))
        .then(methods.replace_strict(COMBINED, default=None, return_dtype=pl.String))
        .otherwise(methods.str.to_uppercase())
    )


def corrected_frag_type(frag_type: pl.Expr, header: pl.Expr) -> pl.Expr:
    """``frag_type`` as the header names it, keeping the stored value when it already agrees."""
    derived = header_frag_type(header)
    agrees = (derived == frag_type) | ((derived == "HCD") & (frag_type == "HCID"))
    return pl.when(derived.is_null() | agrees.fill_null(False)).then(frag_type).otherwise(derived)


def fix_file(path: str, dry_run: bool) -> pl.DataFrame:
    """Rewrite one Parquet file's ``frag_type`` from its headers; return the changes as (old, new, rows)."""
    lazy = pl.scan_parquet(path)
    if not {"header", "frag_type"} <= set(lazy.collect_schema().names()):
        return pl.DataFrame(schema={"old": pl.String, "new": pl.String, "rows": pl.UInt32})
    changes = (
        lazy.select(pl.col("frag_type").alias("old"), corrected_frag_type(pl.col("frag_type"), pl.col("header")).alias("new"))
        .filter(pl.col("old").ne_missing(pl.col("new")))
        .group_by(["old", "new"])
        .agg(pl.len().alias("rows"))
        .collect()
    )
    if changes.height and not dry_run:
        frame = pl.read_parquet(path).with_columns(corrected_frag_type(pl.col("frag_type"), pl.col("header")).alias("frag_type"))
        temporary = f"{path}.tmp"
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    return changes


@app.command()
def main(
    input_dirs: Annotated[List[str], typer.Option("--input-dir", help="Directory searched recursively; repeatable")],
    file_pattern: Annotated[str, typer.Option("--file-pattern", help="Recursive glob under each input directory")] = "**/*.parquet",
    output_file: Annotated[Optional[Path], typer.Option("--output-file", help="CSV of every change: file, old, new, rows")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Report the changes without rewriting files")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Set frag_type from the scan header in every matching Parquet file."""
    configure_script_logging(verbose=verbose)
    files = sorted({path for directory in input_dirs for path in glob.glob(str(Path(directory) / file_pattern), recursive=True)})
    if not files:
        raise typer.BadParameter(f"no files match {file_pattern} under {input_dirs}")

    reports = []
    for path in tqdm(files, desc="frag_type"):
        changes = fix_file(path, dry_run)
        if changes.height:
            reports.append(changes.with_columns(pl.lit(path).alias("file")))
    report = pl.concat(reports) if reports else pl.DataFrame(schema={"old": pl.String, "new": pl.String, "rows": pl.UInt32, "file": pl.String})
    totals = report.group_by(["old", "new"]).agg(pl.col("rows").sum(), pl.len().alias("files")).sort("rows", descending=True)
    action = "would change" if dry_run else "changed"
    logger.info(f"{action} {int(report['rows'].sum() or 0):,} rows in {report['file'].n_unique()} of {len(files)} files")
    for row in totals.iter_rows(named=True):
        logger.info(f"  {row['old']} -> {row['new']}: {row['rows']:,} rows in {row['files']} files")
    if output_file is not None:
        report.select("file", "old", "new", "rows").write_csv(output_file)


if __name__ == "__main__":
    app()
