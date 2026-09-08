# ruff: noqa: T201 - a CLI report: the printed output is the whole point
"""Census the corpus before publishing it to HuggingFace.

Every number in the release manifests comes from here, so that they cannot drift from
the tree. Two phases, both read-only; nothing is written into the data.

**Phase 1, footers.** Parquet metadata only -- no column data is decoded -- across
every file. Establishes per-file rows, bytes and row-groups, the ordered column list
and its dtypes, and therefore:

* whether the reported "empty" files are zero-byte (which makes pyarrow raise, breaking
  a whole `datasets` config) or zero-row-with-schema (harmless but still excluded);
* the real ``-of-NNNNN`` shard totals per split, which must never be guessed -- a wrong
  total makes a complete download look permanently incomplete;
* **whether each config's files share a unifiable schema at all.** A `datasets` config
  is a single unified scan over up to 15,286 files, and the labelled sources are known
  to differ in dtypes, so this decides whether the by-project configs can be published.

**Phase 2, key columns.** Reads only the handful of columns the release contract
depends on. Parquet is columnar, so this touches a few percent of the corpus rather
than all of it. Establishes null counts for the columns a user would try to join on,
whether ``(experiment_name, scan)`` is a usable row key, per-project row counts, and
the filter yield -- applying the real five-condition filter to the unsplit tier and
checking it reproduces the split row count.

Outputs land in ``--out-dir`` as CSV and JSON, plus a summary to stdout. Phase 2 is
scoped with ``--tiers`` so the cheap tiers can answer the important questions first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# No default mount path: it is deployment-specific, and hard-coding one would
# bake a stale internal location into a public repository.
TIERS = ("hcfm", "mcfm", "lcfm")

# Columns the published contract depends on. Phase 2 reads only these.
KEY_COLUMNS = (
    "experiment_name",
    "scan",
    "sequence",
    "unmodified_peptide",
    "normalised_peptide",
    "usi",
    "retention_time",
    "lower_offset",
    "precursor_charge",
    "precursor_mz",
)

# filter_spectra() in scripts/splitting/split_labelled_data.py. Nulls PASS every
# numeric condition, which is easy to get wrong when reimplementing.
MAX_RETENTION_TIME = 10800.0
MAX_LOWER_OFFSET = 300.0
MIN_PRECURSOR_CHARGE = 0
MAX_PRECURSOR_CHARGE = 7
MAX_PRECURSOR_MZ = 2000.0
GLYCO_PATTERN = r"\[[Ii][Nn]:\d+\]"

# A by-project run name containing one of these would materialise a phantom split for
# anyone who runs their own detection instead of using the declared configs.
SPLIT_KEYWORD_PATTERN = r"(?i)(train|test|valid|dev|val)[-._ 0-9]"


@dataclass
class FileRow:
    """One published file, as seen from its parquet footer."""

    path: str
    config: str
    tier: str
    flavour: str
    project: str
    run_name: str
    split: str
    bytes: int
    rows: int
    row_groups: int
    schema_sig: str
    unreadable: str = ""


@dataclass
class ConfigCensus:
    """Everything phase 1 learns about one publishable config."""

    files: int = 0
    bytes: int = 0
    rows: int = 0
    zero_row: list[str] = field(default_factory=list)
    zero_byte: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    schemas: dict[str, int] = field(default_factory=dict)
    columns_first: list[tuple[str, str]] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    projects: set[str] = field(default_factory=set)


def human(n: float) -> str:
    """Format a byte count for humans."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


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
    p.add_argument("--out-dir", type=Path, default=Path("census_out"))
    p.add_argument(
        "--phase", choices=["1", "2", "both"], default="both", help="1=footers, 2=key columns"
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        default=list(TIERS),
        help="tiers for phase 2; phase 1 always covers everything",
    )
    p.add_argument(
        "--progress-every", type=int, default=2000, help="log a progress line every N files"
    )
    return p.parse_args(argv)


def classify(mount: Path) -> list[tuple[str, str, str, Path]]:
    """Return (config, tier, flavour, root) for every directory to census."""
    out = []
    for tier in TIERS:
        for dirname, flavour in ((f"{tier}_splits", "splits"), (tier, "by_project")):
            root = mount / dirname
            if root.is_dir():
                out.append((f"{tier}_{flavour}", tier, flavour, root))
            else:
                print(f"  [warn] absent: {root}")
    return out


def split_of(name: str) -> str:
    """Infer the split from a split-flavour filename, or '' if not applicable."""
    base = name.lower()
    for split, aliases in (
        ("train", ("train",)),
        ("validation", ("validation", "valid", "val", "dev")),
        ("test", ("test", "eval")),
    ):
        for a in aliases:
            if base.startswith(f"{a}_") or base.startswith(f"{a}-"):
                return split
    return ""


def phase1(args: argparse.Namespace) -> tuple[dict[str, ConfigCensus], list[FileRow]]:
    """Footer-only scan of every file."""
    import pyarrow.parquet as pq

    census: dict[str, ConfigCensus] = {}
    rows: list[FileRow] = []
    for config, tier, flavour, root in classify(args.mount):
        c = census.setdefault(config, ConfigCensus())
        files = sorted(f for f in root.rglob("*") if f.is_file())
        print(f"  {config}: {len(files):,} files under {root}", flush=True)
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root)
            size = f.stat().st_size
            project = rel.parts[0] if flavour == "by_project" and len(rel.parts) > 1 else ""
            row = FileRow(
                path=f"{config}/{rel.as_posix()}",
                config=config,
                tier=tier,
                flavour=flavour,
                project=project,
                run_name=f.name,
                split=split_of(f.name) if flavour == "splits" else "",
                bytes=size,
                rows=0,
                row_groups=0,
                schema_sig="",
            )
            c.files += 1
            c.bytes += size
            if project:
                c.projects.add(project)
            if size == 0:
                c.zero_byte.append(row.path)
                rows.append(row)
                continue
            try:
                md = pq.ParquetFile(f).metadata
                sch = md.schema
                cols = [
                    (sch.column(j).name, str(sch.column(j).physical_type))
                    for j in range(md.num_columns)
                ]
                sig = "|".join(f"{n}:{t}" for n, t in cols)
                row.rows, row.row_groups, row.schema_sig = md.num_rows, md.num_row_groups, sig
                c.rows += md.num_rows
                c.schemas[sig] = c.schemas.get(sig, 0) + 1
                if not c.columns_first:
                    c.columns_first = cols
                if md.num_rows == 0:
                    c.zero_row.append(row.path)
                if row.split:
                    c.split_counts[row.split] += 1
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding
                row.unreadable = f"{type(exc).__name__}: {exc}"[:200]
                c.unreadable.append(row.path)
            rows.append(row)
            if i % args.progress_every == 0:
                print(f"    {config}: {i:,}/{len(files):,}", flush=True)
    return census, rows


def report_phase1(census: dict[str, ConfigCensus], rows: list[FileRow], out: Path) -> None:
    """Print the phase-1 verdicts and write the file inventory."""
    print("=" * 78)
    print("PHASE 1 - footers")
    print("=" * 78)
    print(
        f"  {'config':18s} {'files':>8s} {'rows':>16s} {'size':>10s} "
        f"{'schemas':>8s} {'0-row':>6s} {'0-byte':>7s}"
    )
    for config, c in census.items():
        print(
            f"  {config:18s} {c.files:>8,} {c.rows:>16,} {human(c.bytes):>10s} "
            f"{len(c.schemas):>8,} {len(c.zero_row):>6,} {len(c.zero_byte):>7,}"
        )

    print("\n  SCHEMA UNIFIABILITY (a config with >1 distinct schema may fail to load)")
    for config, c in census.items():
        if len(c.schemas) <= 1:
            print(f"    {config:18s} OK - one schema, {len(c.columns_first)} columns")
            continue
        print(f"    {config:18s} *** {len(c.schemas)} DISTINCT SCHEMAS ***")
        ranked = sorted(c.schemas.items(), key=lambda kv: -kv[1])
        base = set(ranked[0][0].split("|"))
        for sig, n in ranked[:4]:
            s = set(sig.split("|"))
            print(f"      {n:>7,} files  +{sorted(s - base)[:4]} -{sorted(base - s)[:4]}")

    print("\n  SHARD TOTALS (use these for -of-NNNNN; never guess)")
    for config, c in census.items():
        if c.split_counts:
            print(
                f"    {config:18s} "
                + "  ".join(f"{k}={v}" for k, v in sorted(c.split_counts.items()))
            )

    print("\n  ACCESSIONS")
    for config, c in census.items():
        if c.projects:
            print(f"    {config:18s} {len(c.projects)} projects")

    for config, c in census.items():
        if c.unreadable:
            print(f"\n  UNREADABLE in {config}: {len(c.unreadable)}")
            for p in c.unreadable[:5]:
                print(f"    {p}")

    import re

    kw = [
        r.path
        for r in rows
        if r.flavour == "by_project" and re.search(SPLIT_KEYWORD_PATTERN, r.run_name)
    ]
    print(f"\n  by_project run names containing a split keyword: {len(kw)}")
    for p in kw[:5]:
        print(f"    {p}")

    basenames: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.flavour == "by_project" and r.tier == "lcfm":
            basenames[r.run_name].append(r.project)
    dupes = {k: v for k, v in basenames.items() if len(v) > 1}
    print(f"  run names appearing in more than one accession (lcfm): {len(dupes)}")
    for k, v in list(dupes.items())[:5]:
        print(f"    {k} in {v[:4]}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "census_files.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "path",
                "config",
                "tier",
                "flavour",
                "project",
                "run_name",
                "split",
                "bytes",
                "rows",
                "row_groups",
                "schema_sig",
                "unreadable",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path,
                    r.config,
                    r.tier,
                    r.flavour,
                    r.project,
                    r.run_name,
                    r.split,
                    r.bytes,
                    r.rows,
                    r.row_groups,
                    r.schema_sig,
                    r.unreadable,
                ]
            )
    summary = {
        cfg: {
            "files": c.files,
            "bytes": c.bytes,
            "rows": c.rows,
            "distinct_schemas": len(c.schemas),
            "zero_row": c.zero_row,
            "zero_byte": c.zero_byte,
            "unreadable": c.unreadable,
            "shard_totals": dict(c.split_counts),
            "projects": sorted(c.projects),
            "columns": c.columns_first,
        }
        for cfg, c in census.items()
    }
    (out / "census_phase1.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out / 'census_files.csv'} and {out / 'census_phase1.json'}")
    # Also echo the summary, because on a batch runner the pod filesystem does not
    # survive the job but the logs do, and these are the numbers the release needs.
    print("\n  --- census_phase1.json ---")
    for line in json.dumps(summary, indent=2).splitlines():
        print(f"  {line}")


def phase2(args: argparse.Namespace) -> None:
    """Key-column scan: null counts, row key usability, per-project rows, filter yield."""
    import polars as pl

    print("=" * 78)
    print("PHASE 2 - key columns")
    print("=" * 78)
    out: dict = {}
    for tier in args.tiers:
        for dirname, flavour in ((f"{tier}_splits", "splits"), (tier, "by_project")):
            root = args.mount / dirname
            if not root.is_dir():
                continue
            files = sorted(f for f in root.rglob("*.parquet") if f.is_file() and f.stat().st_size)
            if not files:
                files = sorted(f for f in root.rglob("*") if f.is_file() and f.stat().st_size)
            config = f"{tier}_{flavour}"
            print(f"  {config}: scanning {len(files):,} files", flush=True)

            nulls: dict[str, int] = defaultdict(int)
            present: dict[str, int] = defaultdict(int)
            total = 0
            kept = 0
            per_project: dict[str, int] = defaultdict(int)
            key_pairs = 0
            key_unique = 0
            for i, f in enumerate(files, 1):
                try:
                    lf = pl.scan_parquet(f)
                    have = [c for c in KEY_COLUMNS if c in lf.collect_schema().names()]
                    df = lf.select(have).collect()
                except Exception as exc:  # noqa: BLE001
                    print(f"    [skip] {f.name}: {type(exc).__name__}")
                    continue
                total += df.height
                for c in have:
                    present[c] += 1
                    nulls[c] += int(df[c].null_count())
                if flavour == "by_project" and {"experiment_name", "scan"} <= set(have):
                    key_pairs += df.height
                    key_unique += df.select(["experiment_name", "scan"]).n_unique()
                    proj = f.relative_to(root).parts[0]
                    per_project[proj] += df.height
                if flavour == "by_project":
                    cond = pl.lit(True)
                    for col, op, thr in (
                        ("retention_time", "le", MAX_RETENTION_TIME),
                        ("lower_offset", "le", MAX_LOWER_OFFSET),
                        ("precursor_charge", "ge", MIN_PRECURSOR_CHARGE),
                        ("precursor_charge", "le", MAX_PRECURSOR_CHARGE),
                        ("precursor_mz", "le", MAX_PRECURSOR_MZ),
                    ):
                        if col not in have:
                            continue
                        c_ = pl.col(col) <= thr if op == "le" else pl.col(col) >= thr
                        cond = cond & (pl.col(col).is_null() | c_)
                    if "sequence" in have:
                        cond = cond & ~(
                            pl.col("sequence")
                            .cast(pl.Utf8, strict=False)
                            .fill_null("")
                            .str.contains(GLYCO_PATTERN)
                        )
                    kept += df.filter(cond).height
                if i % args.progress_every == 0:
                    print(f"    {config}: {i:,}/{len(files):,} rows={total:,}", flush=True)

            print(f"    rows={total:,}")
            print("    null counts in the columns a user would join on:")
            for c in KEY_COLUMNS:
                if present.get(c):
                    pct = 100.0 * nulls[c] / total if total else 0.0
                    flag = "  <-- ALL NULL" if total and nulls[c] == total else ""
                    print(f"      {c:22s} {nulls[c]:>16,} / {total:,}  ({pct:5.1f}%){flag}")
                else:
                    print(f"      {c:22s} column absent")
            if flavour == "by_project":
                print(
                    f"    filter yield: {kept:,} of {total:,} "
                    f"({100.0 * kept / total if total else 0:.3f}%)"
                )
                if key_pairs:
                    print(
                        f"    (experiment_name, scan) unique within file: "
                        f"{key_unique:,} / {key_pairs:,}"
                    )
            out[config] = {
                "rows": total,
                "nulls": dict(nulls),
                "columns_present": sorted(present),
                "filter_kept": kept if flavour == "by_project" else None,
                "per_project_rows": dict(per_project) or None,
                "key_unique_within_file": key_unique or None,
            }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "census_phase2.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out_dir / 'census_phase2.json'}")
    print("\n  --- census_phase2.json ---")
    for line in json.dumps(out, indent=2).splitlines():
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    """Run the requested census phases."""
    args = parse_args(argv)
    if not args.mount.is_dir():
        raise SystemExit(f"error: {args.mount} is not mounted")
    print(f"mount={args.mount}  out={args.out_dir}  phase={args.phase}  tiers={args.tiers}")
    if args.phase in ("1", "both"):
        census, rows = phase1(args)
        report_phase1(census, rows, args.out_dir)
    if args.phase in ("2", "both"):
        phase2(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
