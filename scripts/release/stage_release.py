# ruff: noqa: T201 - a CLI step: the printed staging manifest is the point
r"""Assemble the upload staging tree, so the repo layout exists before anything is sent.

The preparation pass writes the per-project tiers. This adds the other half: the split
partitions, the peptide registry, and the dataset card, arranged exactly as the published
repository will be. Uploading is then a straight copy with no path translation to get
wrong, and the tree can be inspected before a byte leaves the cluster.

The splits are **not rewritten**. They are the artefact the model consumed, and rewriting
them to change a filename would be both wasteful and a needless risk. Instead each file is
**hardlinked** under its published name, which costs no space and no copy time because
both paths live on the same filesystem. Hardlinks specifically, not a symlinked directory:
``Path.glob`` does not descend into those, so the upload would silently send nothing.

Two renames happen here, and both matter to ``datasets``:

* ``valid`` becomes ``validation``. Only the latter is a canonical split name; the former
  is merely an alias, and naming the file ``validation`` is what makes the split load
  under the name the manuscript uses.
* ``train_7.parquet`` becomes ``lcfm-train-00007-of-00293.parquet``. The shard convention
  is what lets a reader see at a glance whether a download is complete, and the tier
  prefix keeps the three tiers distinguishable once downloaded into one directory.

    python scripts/release/stage_release.py \
        --mount "$ROOT" --stage "$ROOT/release_staging" --registry "$ROOT/peptide_registry.parquet"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

TIERS = ("lcfm", "mcfm", "hcfm")

# On disk the splitting pipeline writes "valid"; datasets canonicalises to "validation".
CANONICAL_SPLIT = {"train": "train", "test": "test", "valid": "validation"}

# <split>_<n>.parquet, as written by write_buffer() in the splitting pipeline.
SOURCE_NAME = re.compile(r"^(train|test|valid)_(\d+)\.parquet$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mount", type=Path, required=True, help="directory holding <tier>_splits")
    p.add_argument("--stage", type=Path, required=True, help="staging root to build")
    p.add_argument("--tiers", nargs="+", default=list(TIERS))
    p.add_argument("--registry", type=Path, default=None, help="peptide_registry.parquet")
    p.add_argument("--card", type=Path, default=None, help="dataset card to install as README.md")
    p.add_argument("--dry-run", action="store_true", help="report the plan, create nothing")
    return p.parse_args(argv)


def plan_tier(src_dir: Path, tier: str) -> list[tuple[Path, str]]:
    """Map each split file to its published name.

    Shard indices are assigned from the source file's own number, sorted numerically, so
    the ordering is the pipeline's rather than the filesystem's -- ``train_10`` must not
    sort before ``train_9``.
    """
    by_split: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    unmatched = []
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        m = SOURCE_NAME.match(f.name)
        if not m:
            unmatched.append(f.name)
            continue
        by_split[m.group(1)].append((int(m.group(2)), f))
    if unmatched:
        raise SystemExit(
            f"error: {src_dir} holds files that are not <split>_<n>.parquet: {unmatched[:5]}"
        )

    out: list[tuple[Path, str]] = []
    for split, entries in sorted(by_split.items()):
        entries.sort()
        total = len(entries)
        canonical = CANONICAL_SPLIT[split]
        for i, (_, path) in enumerate(entries):
            out.append((path, f"{tier}-{canonical}-{i:05d}-of-{total:05d}.parquet"))
    return out


def link(src: Path, dst: Path) -> str:
    """Hardlink *src* to *dst*, reporting which mechanism was used."""
    if dst.exists():
        if dst.samefile(src):
            return "present"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        raise SystemExit(
            f"error: cannot hardlink {src} -> {dst}: {exc}\n"
            "Both paths must be on one filesystem. Do not substitute a symlinked "
            "directory: Path.glob will not descend into it and the upload would send "
            "nothing."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """Build the staging tree."""
    args = parse_args(argv)
    total_files = total_bytes = 0

    for tier in args.tiers:
        src_dir = args.mount / f"{tier}_splits"
        if not src_dir.is_dir():
            raise SystemExit(f"error: {src_dir} is not a directory")
        dst_dir = args.stage / "splits" / tier
        pairs = plan_tier(src_dir, tier)
        counts: dict[str, int] = defaultdict(int)
        made = 0
        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
        for src, name in pairs:
            counts[name.split("-")[1]] += 1
            total_bytes += src.stat().st_size
            if not args.dry_run and link(src, dst_dir / name) == "hardlink":
                made += 1
        total_files += len(pairs)
        shape = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        suffix = "" if args.dry_run else f", {made} linked"
        print(f"  splits/{tier}: {len(pairs):>4} files ({shape}){suffix}")
        if pairs:
            print(f"      first: {pairs[0][1]}")
            print(f"      last : {pairs[-1][1]}")

    if args.registry:
        if not args.registry.is_file():
            raise SystemExit(f"error: --registry {args.registry} is not a file")
        if not args.dry_run:
            args.stage.mkdir(parents=True, exist_ok=True)
            link(args.registry, args.stage / args.registry.name)
        total_files += 1
        total_bytes += args.registry.stat().st_size
        print(f"  {args.registry.name}: linked at the staging root")

    if args.card:
        if not args.card.is_file():
            raise SystemExit(f"error: --card {args.card} is not a file")
        if not args.dry_run:
            args.stage.mkdir(parents=True, exist_ok=True)
            (args.stage / "README.md").write_text(args.card.read_text())
        print(f"  README.md: installed from {args.card}")

    gb = total_bytes / 1024**3
    print(
        f"\n  staged {total_files:,} split files / {gb:,.1f} GB"
        f"{' (dry run: nothing created)' if args.dry_run else ' (hardlinked: no space used)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
