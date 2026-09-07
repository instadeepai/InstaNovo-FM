# ruff: noqa: T201 - a CLI step: the printed table is the output
r"""Per-split row counts for every tier, read from parquet footers.

The published 80/10/10 ratio is over *peptides*: the registry assigns each peptide to one
split, and every spectrum of that peptide follows it. Peptides differ in how many spectra
they have, so the row proportions need not match the peptide proportions -- and in HCFM they
do not (67/5/28 against a nominal 80/10/10). This measures all three tiers so we can say
whether that skew is tier-specific or present throughout.

Footers only: ``num_rows`` lives in each file's metadata, so this reads kilobytes per file
rather than gigabytes. Counting 181 million LCFM rows by loading them would move ~467 GB to
learn a number the files already state.

    python scripts/release/audit/split_row_counts.py --mount "$ROOT" --out counts.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TIERS = ("hcfm", "mcfm", "lcfm")

# The splitting pipeline writes <split>_<n>.parquet with "valid"; the release renames it.
SOURCE_NAME = re.compile(r"^(train|test|valid)_(\d+)\.parquet$")
CANONICAL = {"train": "train", "test": "test", "valid": "validation"}
ORDER = ("train", "validation", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mount", type=Path, required=True, help="directory holding <tier>_splits")
    p.add_argument("--tiers", nargs="+", default=list(TIERS))
    p.add_argument("--registry", type=Path, default=None, help="peptide_registry.parquet")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def tier_counts(src: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (rows per canonical split, files per canonical split) from footers."""
    import pyarrow.parquet as pq

    rows: dict[str, int] = defaultdict(int)
    files: dict[str, int] = defaultdict(int)
    for f in sorted(src.iterdir()):
        m = SOURCE_NAME.match(f.name)
        if not m:
            continue
        split = CANONICAL[m.group(1)]
        rows[split] += pq.ParquetFile(f).metadata.num_rows
        files[split] += 1
    return dict(rows), dict(files)


def main(argv: list[str] | None = None) -> int:
    """Report per-split row counts and shares for each tier."""
    args = parse_args(argv)
    result: dict[str, dict] = {}

    for tier in args.tiers:
        src = args.mount / f"{tier}_splits"
        if not src.is_dir():
            print(f"  {tier}: {src} absent, skipping")
            continue
        rows, files = tier_counts(src)
        total = sum(rows.values())
        result[tier] = {"rows": rows, "files": files, "total": total}
        print(f"\n  {tier}_splits — {total:,} rows in {sum(files.values())} files")
        for split in ORDER:
            n = rows.get(split, 0)
            share = 100.0 * n / total if total else 0.0
            print(f"    {split:<11} {n:>13,}  {share:5.2f}%  ({files.get(split, 0)} files)")

    if args.registry and args.registry.is_file():
        import polars as pl

        reg = pl.read_parquet(args.registry, columns=["split"])
        counts = dict(reg.group_by("split").len().iter_rows())
        total = sum(counts.values())
        result["registry_peptides"] = {"counts": counts, "total": total}
        print(f"\n  registry — {total:,} peptides (this is what 80/10/10 describes)")
        for split in ORDER:
            n = counts.get(split, 0)
            print(f"    {split:<11} {n:>13,}  {100.0 * n / total:5.2f}%")

        print("\n  peptide share vs PSM share, per tier:")
        header = "    tier     " + "".join(f"{s:>26}" for s in ORDER)
        print(header)
        for tier in args.tiers:
            if tier not in result:
                continue
            cells = ""
            for split in ORDER:
                pep = 100.0 * counts.get(split, 0) / total
                psm = 100.0 * result[tier]["rows"].get(split, 0) / result[tier]["total"]
                cells += f"{f'{pep:.2f}% -> {psm:.2f}%':>26}"
            print(f"    {tier:<9}{cells}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
