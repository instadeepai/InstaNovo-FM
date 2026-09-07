# ruff: noqa: T201 - a CLI report: the printed cross-tab is the whole point
"""Cross-tabulate precursor charge against acquisition mode.

The filter audit found 6,521,868 spectra with ``precursor_charge == 0`` in the LCFM
training split -- 3.6% of it -- plus 680,200 in MCFM and 231,813 in HCFM. The code keeps
them deliberately (``min_precursor_charge = 0``, compared with ``>=``), so the question
is not whether the pipeline is behaving as written but whether a charge of 0 is
*meaningful*.

The hypothesis to test: these are DIA spectra. In data-independent acquisition the
precursor is an isolation window rather than a selected ion, so no single charge state
is assigned and 0 is the natural encoding of "not applicable". If that holds, charge 0
is an expected property of a corpus spanning both acquisition modes, and the manuscript
can say so in a clause. If it does not hold -- if charge 0 turns up in DDA, where a
charge state should have been determined -- then it points at upstream metadata loss and
deserves more than a clause.

Reports, per config: the acquisition distribution of charge-0 rows, the charge
distribution within each acquisition mode, and the charge-0 rate per mode. Reads only
``precursor_charge`` and ``acquisition``, so it touches a small fraction of the corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# No default mount path: it is deployment-specific, and hard-coding one would
# bake a stale internal location into a public repository.
RELEASE_CONFIGS: dict[str, str] = {
    "hcfm_splits": "hcfm_splits",
    "mcfm_splits": "mcfm_splits",
    "lcfm_splits": "lcfm_splits",
    "hcfm_by_project": "hcfm",
    "mcfm_by_project": "mcfm",
    "lcfm_by_project": "lcfm",
}
COLS = ("precursor_charge", "acquisition")


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
    p.add_argument(
        "--configs", nargs="+", default=list(RELEASE_CONFIGS), choices=list(RELEASE_CONFIGS)
    )
    p.add_argument("--out-dir", type=Path, default=Path("crosstab_out"))
    p.add_argument("--progress-every", type=int, default=2000)
    return p.parse_args(argv)


def scan(files: list[Path], every: int, label: str) -> tuple[dict[tuple[str, int], int], list[str]]:
    """Return {(acquisition, charge): rows} and any skipped files."""
    import polars as pl

    counts: dict[tuple[str, int], int] = defaultdict(int)
    skipped: list[str] = []
    seen = 0
    for i, f in enumerate(files, 1):
        try:
            lf = pl.scan_parquet(f)
            names = lf.collect_schema().names()
            missing = [c for c in COLS if c not in names]
            if missing:
                skipped.append(f"{f.name}: missing {missing}")
                continue
            df = lf.select(
                pl.col("acquisition").cast(pl.Utf8, strict=False).fill_null("(null)"),
                pl.col("precursor_charge").cast(pl.Int64, strict=False),
            ).collect()
        except Exception as exc:  # noqa: BLE001 - a bad file is a finding
            skipped.append(f"{f.name}: {type(exc).__name__}: {exc}"[:160])
            continue
        seen += df.height
        grouped = df.group_by(["acquisition", "precursor_charge"]).len()
        for acq, ch, n in zip(
            grouped["acquisition"].to_list(),
            grouped["precursor_charge"].to_list(),
            grouped["len"].to_list(),
            strict=True,
        ):
            counts[(acq, -1 if ch is None else int(ch))] += int(n)
        if i % every == 0:
            print(f"    {label}: {i:,}/{len(files):,} rows={seen:,}", flush=True)
    return counts, skipped


def report(label: str, counts: dict[tuple[str, int], int], skipped: list[str]) -> dict:
    """Print the cross-tab for one config and return it as data."""
    total = sum(counts.values())
    acqs = sorted({a for a, _ in counts})
    charges = sorted({c for _, c in counts})
    print(f"\n  === {label} === rows={total:,}")
    if skipped:
        print(f"    SKIPPED {len(skipped)} file(s) — counts incomplete: {skipped[:3]}")

    hdr = (
        "    "
        + f"{'acquisition':22s}"
        + "".join(f"{('null' if c == -1 else c):>14}" for c in charges)
        + f"{'total':>16}"
    )
    print(hdr)
    for a in acqs:
        row = [counts.get((a, c), 0) for c in charges]
        print("    " + f"{a[:22]:22s}" + "".join(f"{n:>14,}" for n in row) + f"{sum(row):>16,}")

    zero = {a: counts.get((a, 0), 0) for a in acqs}
    z_total = sum(zero.values())
    print(
        f"\n    charge-0 rows: {z_total:,} "
        f"({100.0 * z_total / total if total else 0:.2f}% of config)"
    )
    if z_total:
        print("    where they live:")
        for a in sorted(acqs, key=lambda x: -zero[x]):
            if zero[a]:
                per_mode = sum(counts.get((a, c), 0) for c in charges)
                print(
                    f"      {a[:22]:22s} {zero[a]:>14,}  "
                    f"({100.0 * zero[a] / z_total:5.1f}% of charge-0; "
                    f"{100.0 * zero[a] / per_mode if per_mode else 0:5.1f}% of this mode)"
                )
    return {
        "rows": total,
        "by_acquisition_charge": {f"{a}|{c}": n for (a, c), n in sorted(counts.items())},
        "charge_zero_total": z_total,
        "charge_zero_by_acquisition": zero,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    """Cross-tabulate charge against acquisition across the release configs."""
    args = parse_args(argv)
    if not args.mount.is_dir():
        raise SystemExit(f"error: {args.mount} is not a directory")

    out: dict = {}
    for cfg in args.configs:
        root = args.mount / RELEASE_CONFIGS[cfg]
        if not root.is_dir():
            print(f"  [warn] absent: {cfg} -> {root}")
            continue
        files = sorted(f for f in root.rglob("*") if f.is_file() and f.stat().st_size)
        print(f"  {cfg}: {len(files):,} files")
        counts, skipped = scan(files, args.progress_every, cfg)
        out[cfg] = report(cfg, counts, skipped)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "charge_acquisition.json").write_text(json.dumps(out, indent=2))
    with (args.out_dir / "charge_acquisition.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "acquisition", "precursor_charge", "rows"])
        for cfg, d in out.items():
            for key, n in d["by_acquisition_charge"].items():
                acq, ch = key.rsplit("|", 1)
                w.writerow([cfg, acq, ch, n])
    print(f"\n  wrote {args.out_dir}/charge_acquisition.{{json,csv}}")
    print("\n  --- summary ---")
    for line in json.dumps(
        {
            k: {kk: vv for kk, vv in v.items() if kk != "by_acquisition_charge"}
            for k, v in out.items()
        },
        indent=2,
    ).splitlines():
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
