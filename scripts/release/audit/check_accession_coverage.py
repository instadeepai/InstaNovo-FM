# ruff: noqa: T201 - a CLI check: the printed diff is the output
r"""Diff the accessions in the published tiers against the paper's accession table.

The manuscript states 92 accessions in the assembled corpus and Supplementary Table S1
lists them; the published per-project tiers contain 82 directories. Both can be true --
the 92 describe the assembled corpus, which includes the unlabelled ACFM tier, while a
published tier only contains accessions that contributed PSMs surviving that tier's
confidence threshold. But nobody has checked, and a table captioned "all 92 accessions"
sitting beside data showing 82 invites the question.

This names the difference either way: which of the 92 are absent from each tier, and
whether any published directory is missing from the table (which would be the more
serious finding -- data whose provenance the paper does not record).

    python scripts/release/audit/check_accession_coverage.py \
        --mount "$ROOT" --table-s1 table_s1_accessions.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TIERS = ("hcfm", "mcfm", "lcfm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mount", type=Path, required=True, help="directory holding the tiers")
    p.add_argument("--tiers", nargs="+", default=list(TIERS))
    p.add_argument(
        "--table-s1", type=Path, required=True, help="one accession per line, from Table S1"
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report the accession diff per tier."""
    args = parse_args(argv)
    paper = {a.strip() for a in args.table_s1.read_text().split() if a.strip()}
    print(f"  Table S1 lists {len(paper)} accessions")

    # holds both the accession list and the per-tier dicts, hence Any
    result: dict[str, Any] = {"table_s1": sorted(paper)}
    per_tier: dict[str, set[str]] = {}

    for tier in args.tiers:
        root = args.mount / tier
        if not root.is_dir():
            print(f"  {tier}: {root} absent, skipping")
            continue
        dirs = {d.name for d in root.iterdir() if d.is_dir()}
        per_tier[tier] = dirs
        missing = sorted(paper - dirs)
        unlisted = sorted(dirs - paper)
        result[tier] = {
            "published": len(dirs),
            "absent_from_tier": missing,
            "not_in_table_s1": unlisted,
        }
        print(f"\n  {tier}: {len(dirs)} accession directories")
        print(f"    in Table S1 but not in this tier ({len(missing)}): {missing}")
        if unlisted:
            print(f"    *** in this tier but NOT in Table S1 ({len(unlisted)}): {unlisted}")
            print("    This is the serious direction: published data the paper does not list.")
        else:
            print("    every published accession appears in Table S1")

    if len(per_tier) > 1:
        tiers = list(per_tier)
        common = set.intersection(*per_tier.values())
        print(f"\n  accessions present in all {len(tiers)} tiers: {len(common)}")
        for tier in tiers:
            extra = sorted(per_tier[tier] - common)
            if extra:
                print(f"    only in some tiers — {tier} additionally has: {extra}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
