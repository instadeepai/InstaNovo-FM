# ruff: noqa: T201 - a CLI audit: the printed report is the whole point
r"""Audit whether the published splits actually satisfy the quality filters.

Six places in the repository document these filters and they disagree. Correcting the
prose is only sufficient if the *data* already conforms; if it does not, the splits
contain rows the filters were supposed to remove, and that is a data problem rather
than a documentation one.

``filter_spectra()`` in ``scripts/splitting/split_labelled_data.py`` is the truth:

    retention_time   <= 10800.0   nulls pass
    lower_offset     <= 300.0     nulls pass
    precursor_charge >= 0         nulls pass
    precursor_charge <= 7         nulls pass
    precursor_mz     <= 2000.0    nulls pass
    NOT sequence ~ r"\\[[Ii][Nn]:\\d+\\]"        (glyco-modified PSMs)

The test: apply all six conditions to the ``<tier>_splits`` data, which has supposedly
already been filtered. **Every row should survive.** Any shortfall names the condition
that was not applied, and per-condition counts localise it.

Two questions this settles that a row-count reconciliation cannot:

* **Does charge 0 occur?** The code keeps it (``>=``), two READMEs say it is excluded
  (``1-7``, ``> 0``). Those give identical row counts when no charge-0 row exists, so
  only a direct count distinguishes "docs wrong" from "code wrong".
* **How much passed on nulls?** ``_nullable_filter`` lets nulls through every numeric
  condition, so a row with a null precursor m/z is in the training data by design.
  Anyone reimplementing the filter who drops nulls gets a different dataset.

Also counts the same things in the unsplit tiers, to show what the filter removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# No default mount path: it is deployment-specific, and hard-coding one would
# bake a stale internal location into a public repository.
TIERS = ("hcfm", "mcfm", "lcfm")
GLYCO = r"\[[Ii][Nn]:\d+\]"

CONDITIONS = (
    ("retention_time_gt_10800", "retention_time", "gt", 10800.0),
    ("lower_offset_gt_300", "lower_offset", "gt", 300.0),
    ("charge_lt_0", "precursor_charge", "lt", 0),
    ("charge_gt_7", "precursor_charge", "gt", 7),
    ("precursor_mz_gt_2000", "precursor_mz", "gt", 2000.0),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mount",
        type=Path,
        required=True,
        help="root holding the tier directories (required; no default)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("conformance_out"))
    p.add_argument("--tiers", nargs="+", default=list(TIERS))
    p.add_argument(
        "--flavours",
        nargs="+",
        default=["splits", "by_project"],
        help="splits is the conformance test; by_project shows what was removed",
    )
    p.add_argument("--progress-every", type=int, default=2000)
    return p.parse_args(argv)


def scan(root: Path, progress_every: int, label: str) -> dict:
    """Count rows violating each condition, plus nulls and charge-0."""
    import polars as pl

    files = sorted(f for f in root.rglob("*") if f.is_file() and f.stat().st_size)
    print(f"  {label}: {len(files):,} files", flush=True)

    total = 0
    viol: dict[str, int] = defaultdict(int)
    nulls: dict[str, int] = defaultdict(int)
    charge_zero = 0
    glyco_rows = 0
    absent: set[str] = set()
    skipped: list[str] = []

    # dict.fromkeys dedupes while keeping order: precursor_charge appears in two
    # conditions, and selecting it twice raises polars DuplicateError.
    wanted = list(dict.fromkeys([c[1] for c in CONDITIONS] + ["sequence"]))
    for i, f in enumerate(files, 1):
        try:
            lf = pl.scan_parquet(f)
            have = set(lf.collect_schema().names())
            df = lf.select([c for c in wanted if c in have]).collect()
        except Exception as exc:  # noqa: BLE001 - a bad file is a finding
            # A skipped file must never produce a passing verdict: zero rows read
            # trivially satisfies every condition, which is how a silent failure
            # dresses itself up as conformance.
            skipped.append(f"{f.name}: {type(exc).__name__}: {exc}"[:200])
            print(f"    [skip] {f.name}: {type(exc).__name__}")
            continue
        total += df.height
        absent |= {c for c in wanted if c not in have}

        for name, col, op, thr in CONDITIONS:
            if col not in df.columns:
                continue
            expr = pl.col(col) > thr if op == "gt" else pl.col(col) < thr
            viol[name] += int(df.filter(expr.fill_null(False)).height)

        for col in wanted:
            if col in df.columns:
                nulls[col] += int(df[col].null_count())

        if "precursor_charge" in df.columns:
            charge_zero += int(df.filter(pl.col("precursor_charge") == 0).height)
        if "sequence" in df.columns:
            glyco_rows += int(
                df.filter(
                    pl.col("sequence").cast(pl.Utf8, strict=False).fill_null("").str.contains(GLYCO)
                ).height
            )
        if i % progress_every == 0:
            print(f"    {label}: {i:,}/{len(files):,} rows={total:,}", flush=True)

    return {
        "rows": total,
        "violations": dict(viol),
        "glyco_rows": glyco_rows,
        "charge_zero_rows": charge_zero,
        "nulls": dict(nulls),
        "columns_absent": sorted(absent),
        "files": len(files),
        "files_skipped": skipped,
    }


def report(label: str, r: dict, is_splits: bool) -> bool:
    """Print one config's findings. Returns True if it conforms."""
    total = r["rows"]
    bad = sum(r["violations"].values()) + r["glyco_rows"]
    print(f"\n  {label}  rows={total:,}")
    for name, _, _, _ in CONDITIONS:
        n = r["violations"].get(name, 0)
        mark = "  <-- VIOLATION" if (is_splits and n) else ""
        print(f"    {name:26s} {n:>14,}{mark}")
    mark = "  <-- VIOLATION" if (is_splits and r["glyco_rows"]) else ""
    print(f"    {'glyco_sequences':26s} {r['glyco_rows']:>14,}{mark}")
    print(
        f"    {'charge == 0':26s} {r['charge_zero_rows']:>14,}"
        f"{'   (kept by design: charge 0 marks DIA spectra)' if r['charge_zero_rows'] else ''}"
    )
    print("    nulls that passed on null-tolerance:")
    for col, n in sorted(r["nulls"].items()):
        if n:
            print(f"      {col:24s} {n:>14,}  ({100.0 * n / total if total else 0:.4f}%)")
    if r["columns_absent"]:
        print(f"    columns absent somewhere: {r['columns_absent']}")
    if r["files_skipped"]:
        print(f"    UNREADABLE: {len(r['files_skipped'])} file(s) skipped — verdict withheld")
        for x in r["files_skipped"][:5]:
            print(f"      {x}")
    if is_splits:
        if r["files_skipped"]:
            verdict = "INCONCLUSIVE (files skipped)"
        elif total == 0:
            verdict = "INCONCLUSIVE (no rows read)"
        elif bad:
            verdict = f"{bad:,} NON-CONFORMING ROWS"
        else:
            verdict = "CONFORMS"
        print(f"    -> {verdict}")
        return verdict == "CONFORMS"
    return True


def main(argv: list[str] | None = None) -> int:
    """Audit filter conformance across the requested tiers and flavours."""
    args = parse_args(argv)
    if not args.mount.is_dir():
        raise SystemExit(f"error: {args.mount} is not mounted")
    print(f"mount={args.mount} tiers={args.tiers} flavours={args.flavours}")

    out: dict = {}
    conforms = True
    for tier in args.tiers:
        for flavour in args.flavours:
            dirname = f"{tier}_splits" if flavour == "splits" else tier
            root = args.mount / dirname
            if not root.is_dir():
                print(f"  [warn] absent: {root}")
                continue
            label = f"{tier}_{flavour}"
            r = scan(root, args.progress_every, label)
            out[label] = r
            ok = report(label, r, flavour == "splits")
            if flavour == "splits":
                conforms = conforms and ok

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "conformance.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 78)
    print(
        "VERDICT: splits conform to filter_spectra()"
        if conforms
        else "VERDICT: splits contain rows the filters should have removed"
    )
    print("=" * 78)
    print("\n--- conformance.json ---")
    for line in json.dumps(out, indent=2).splitlines():
        print(f"  {line}")
    return 0 if conforms else 1


if __name__ == "__main__":
    sys.exit(main())
