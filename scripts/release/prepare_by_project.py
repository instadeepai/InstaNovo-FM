# ruff: noqa: T201 - a CLI step: the printed progress and the divergence report are
# the output, and on a batch pod the log is the only thing that outlives the job.
r"""Prepare the per-project (unsplit) tiers so they can be published as loadable configs.

The split partitions are published exactly as the model consumed them and are never
rewritten. The per-project tiers are different: they are the raw per-accession output of
tier construction, and their files do not share a schema. A HuggingFace ``datasets``
config is a single unified scan over every file in it, so as they stand these tiers
cannot be loaded at all. Three divergences cause it:

* ``collision_energy`` is stored as a string in some files and a float in others.
* ``auc_intensity`` is an integer in a few files and a float elsewhere.
* ``retention`` is absent from some files entirely.

Only the third is handled by the splitting pipeline's ``normalise_dataframe_schema``,
which adds missing columns but does not cast the ones already present. So this script
casts explicitly, and refuses to do it blindly: a value that will not parse is reported,
never quietly replaced with null. Losing a real measurement to a silent cast across
thousands of files is the failure this script exists to avoid.

It also does three smaller things the release needs:

* adds ``registry_key``, the column that actually joins to the peptide registry --
  ``unmodified_peptide`` with ``I`` collapsed to ``L``. Note the direction: the registry
  uses ``I``->``L`` and its ``peptide`` column contains no ``I`` at all.
* drops ``normalised_peptide``, which is all-null where it exists and absent elsewhere,
  and ``isolation_target_old``, a pre-inference backup nothing reads. Both are traps:
  ``normalised_peptide`` is the obvious join key by name and would match nothing.
* skips zero-row files, listing them, because an empty high-confidence run is a finding
  about that run rather than a defect to hide.

Run ``inspect`` before ``run``. Inspect reads schemas and the divergent columns only, so
it is cheap, and it answers the question ``run`` needs settled first: what is actually in
those string values.

    python scripts/release/prepare_by_project.py inspect --mount "$ROOT" --out inspect.json
    python scripts/release/prepare_by_project.py run --mount "$ROOT" --out-dir "$STAGE"

**Compatibility.** This runs on a slim image that pins an older polars than a dev
checkout is likely to have, so test it against the pinned version, not the local one::

    uv venv /tmp/v112 --python 3.11
    uv pip install --python /tmp/v112/bin/python "polars==<pinned>" pytest
    /tmp/v112/bin/python -m pytest tests/test_prepare_by_project.py

A run has already been lost to this: a keyword argument added to silence a local
deprecation warning did not exist in the pinned polars, and the job died partway
through the first tier. Prefer the plainest API that works on both.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

TIERS = ("lcfm", "mcfm", "hcfm")

# Mirrors REFERENCE_SCHEMA in scripts/splitting/split_labelled_data.py. Duplicated rather
# than imported because that module pulls in huggingface_hub, which the slim image used
# to run this does not carry; tests assert the two stay identical.
TARGET_SCHEMA: dict[str, pl.DataType] = {
    "usi": pl.String,
    "index": pl.Int64,
    "scan": pl.String,
    "header": pl.String,
    "retention_time": pl.Float64,
    "frag_type": pl.String,
    "acquisition": pl.String,
    "collision_energy": pl.Float64,
    "isolation_target": pl.Float64,
    "precursor_mz": pl.Float64,
    "precursor_charge": pl.Int64,
    "precursor_intensity": pl.Float64,
    "lower_offset": pl.Float64,
    "upper_offset": pl.Float64,
    "mz_array": pl.List(pl.Float64),
    "intensity_array": pl.List(pl.Float32),
    "scale_factor": pl.Float32,
    "peptide_observed_mz": pl.Float64,
    "peptide_calc_mz": pl.Float64,
    "delta_mass": pl.Float64,
    "retention": pl.Float64,
    "expectation": pl.Float64,
    "hyperscore": pl.Float64,
    "nextscore": pl.Float64,
    "probability": pl.Float64,
    "auc_intensity": pl.Float64,
    "protein": pl.String,
    "experiment_name": pl.String,
    "unmodified_peptide": pl.String,
    "sequence": pl.String,
}

# Dropped on the way out. See the module docstring for why each one goes.
DROP_COLUMNS = ("normalised_peptide", "isolation_target_old")

# The column that joins to the peptide registry, and the source it is derived from.
REGISTRY_KEY = "registry_key"
REGISTRY_KEY_SOURCE = "unmodified_peptide"

# Columns whose stored type is known to vary between files.
DIVERGENT = ("collision_energy", "auc_intensity", "retention")

# Casts that lose precision rather than nullability. A Float64 value narrowed to Float32
# stays non-null, so the parse-failure count above cannot see it -- but the number does
# change. These are counted separately and block a run just the same.
NARROWING = {
    (pl.Float64, pl.Float32),
    (pl.Int64, pl.Float32),
}


def _narrows(src: pl.DataType, dst: pl.DataType) -> bool:
    """True if casting *src* to *dst* can change a value without nulling it."""
    if isinstance(src, pl.List) and isinstance(dst, pl.List):
        return _narrows(src.inner, dst.inner)
    return (src, dst) in NARROWING


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    ins = sub.add_parser("inspect", help="report schemas and what the divergent columns hold")
    ins.add_argument("--mount", type=Path, required=True, help="directory holding the tiers")
    ins.add_argument("--tiers", nargs="+", default=list(TIERS))
    ins.add_argument("--out", type=Path, required=True, help="JSON to write")
    ins.add_argument(
        "--max-samples", type=int, default=40, help="distinct sample values to keep per column"
    )

    run = sub.add_parser("run", help="rewrite the tiers with a unified schema")
    run.add_argument("--mount", type=Path, required=True)
    run.add_argument("--tiers", nargs="+", default=list(TIERS))
    run.add_argument("--out-dir", type=Path, required=True, help="staging root to write into")
    run.add_argument(
        "--allow-cast-loss",
        action="store_true",
        help="proceed even when a value fails to parse (it becomes null). Off by default.",
    )
    run.add_argument("--overwrite", action="store_true", help="rewrite outputs that already exist")
    run.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all)")
    run.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="where empty_runs.csv goes (default: <out-dir>/manifests)",
    )
    return p.parse_args(argv)


def tier_files(mount: Path, tier: str) -> list[Path]:
    """Every parquet under a tier, sorted for a deterministic run order."""
    return sorted((mount / tier).glob("**/*.parquet"))


def relative_output(src: Path, mount: Path, tier: str, out_dir: Path) -> Path:
    """Mirror ``<tier>/<accession>/<run>.parquet`` under the staging root."""
    return out_dir / tier / src.relative_to(mount / tier)


def cast_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Cast to TARGET_SCHEMA, returning the frame and per-column parse-failure counts.

    A failure is a value that is non-null before the cast and null after it. Counting
    them is the whole point: a permissive cast that reports nothing is how a column of
    real measurements silently becomes a column of nulls.
    """
    losses: dict[str, int] = {}
    for name, dtype in TARGET_SCHEMA.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(name))
            continue
        if df.schema[name] == dtype:
            continue
        src = df.schema[name]
        before = df[name].is_not_null().sum()
        narrowing = _narrows(src, dtype)
        original = df[name] if narrowing else None
        df = df.with_columns(pl.col(name).cast(dtype, strict=False).alias(name))
        after = df[name].is_not_null().sum()
        if before != after:
            losses[name] = int(before - after)
        if narrowing:
            # Round-trip back to the wider type and count values that moved.
            restored = df[name].cast(src, strict=False)
            changed = _count_changed(original, restored)
            if changed:
                losses[f"{name} (precision)"] = changed
    return df, losses


def _count_changed(before: pl.Series, after: pl.Series) -> int:
    """Number of elements whose value differs, descending into list columns."""
    if isinstance(before.dtype, pl.List):
        # Plain explode, deliberately: the empty_as_null kwarg does not exist in the
        # polars the runner image pins, and drop_nulls makes both of its semantics
        # identical here anyway -- an empty spectrum contributes nothing either way.
        b = before.explode().drop_nulls()
        a = after.explode().drop_nulls()
        if b.len() != a.len():
            return int(max(b.len(), a.len()))
        return int((b != a).sum())
    return int((before != after).sum())


def prepare_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply the full transformation to one frame."""
    df = df.drop([c for c in DROP_COLUMNS if c in df.columns])
    df, losses = cast_frame(df)
    df = df.select(list(TARGET_SCHEMA.keys()))
    df = df.with_columns(pl.col(REGISTRY_KEY_SOURCE).str.replace_all("I", "L").alias(REGISTRY_KEY))
    return df, losses


def do_inspect(args: argparse.Namespace) -> int:
    """Report what the divergent columns hold, without rewriting anything."""
    report: dict[str, dict] = {}
    for tier in args.tiers:
        files = tier_files(args.mount, tier)
        print(f"  {tier}: {len(files):,} files", flush=True)
        dtypes: dict[str, Counter] = defaultdict(Counter)
        samples: dict[str, set] = defaultdict(set)
        empty: list[str] = []
        unexpected: Counter = Counter()
        for n, f in enumerate(files, 1):
            try:
                lf = pl.scan_parquet(f)
                schema = lf.collect_schema()
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop the survey
                unexpected[f"unreadable: {type(exc).__name__}"] += 1
                continue
            for name, dt in schema.items():
                dtypes[name][str(dt)] += 1
            for name in DIVERGENT:
                if name in schema and schema[name] != TARGET_SCHEMA[name]:
                    if len(samples[name]) < args.max_samples:
                        vals = lf.select(pl.col(name)).drop_nulls().unique().head(8).collect()
                        samples[name].update(str(v) for v in vals[name].to_list())
            if n % 2000 == 0:
                print(f"    {tier}: {n:,}/{len(files):,}", flush=True)
        # zero-row files, counted from metadata rather than by reading
        for f in files:
            try:
                if pl.scan_parquet(f).select(pl.len()).collect().item() == 0:
                    empty.append(str(f.relative_to(args.mount)))
            except Exception:  # noqa: BLE001, S110
                pass
        report[tier] = {
            "files": len(files),
            "column_dtypes": {k: dict(v) for k, v in sorted(dtypes.items())},
            "divergent_samples": {k: sorted(v) for k, v in samples.items()},
            "empty_files": empty,
            "problems": dict(unexpected),
        }
        for name, counts in sorted(dtypes.items()):
            if len(counts) > 1:
                shown = ", ".join(f"{dt} x{n:,}" for dt, n in counts.most_common())
                print(
                    f"    DIVERGENT {name}: {shown}   target={TARGET_SCHEMA.get(name)}", flush=True
                )
        print(f"    zero-row files: {len(empty):,}", flush=True)
        for name, vals in samples.items():
            print(f"    {name} non-conforming sample: {sorted(vals)[:12]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n  wrote {args.out}")
    return 0


def do_run(args: argparse.Namespace) -> int:
    """Rewrite each tier into the staging root with one unified schema."""
    totals: Counter = Counter()
    losses_total: Counter = Counter()
    empty_rows: list[tuple[str, str]] = []
    for tier in args.tiers:
        files = tier_files(args.mount, tier)
        print(f"  {tier}: {len(files):,} files", flush=True)
        for n, src in enumerate(files, 1):
            if args.limit and totals["written"] + totals["skipped_empty"] >= args.limit:
                break
            dst = relative_output(src, args.mount, tier, args.out_dir)
            if dst.exists() and not args.overwrite:
                totals["already_present"] += 1
                continue
            df = pl.read_parquet(src)
            if df.height == 0:
                empty_rows.append((tier, str(src.relative_to(args.mount / tier))))
                totals["skipped_empty"] += 1
                continue
            out, losses = prepare_frame(df)
            if losses and not args.allow_cast_loss:
                print(
                    f"\n  STOPPING: {src} has values that will not parse: {losses}."
                    f"\n  Inspect them before deciding. Re-run with --allow-cast-loss only"
                    f"\n  once you know those values are not measurements worth keeping.",
                    file=sys.stderr,
                )
                return 2
            for name, count in losses.items():
                losses_total[name] += count
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.write_parquet(dst)
            totals["written"] += 1
            totals["rows"] += out.height
            if n % 500 == 0:
                print(f"    {tier}: {n:,}/{len(files):,} rows={totals['rows']:,}", flush=True)

    manifest_dir = args.manifest_dir or (args.out_dir / "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "empty_runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "path"])
        w.writerows(empty_rows)

    print(f"\n  written          : {totals['written']:,}")
    print(f"  rows             : {totals['rows']:,}")
    print(f"  zero-row skipped : {totals['skipped_empty']:,}")
    print(f"  already present  : {totals['already_present']:,}")
    if losses_total:
        print(f"  PARSE FAILURES   : {dict(losses_total)}")
    print(f"  empty-run list   : {manifest_dir / 'empty_runs.csv'}")

    # The acceptance test. A config is one unified scan over every file in it, so
    # scanning a whole tier is exactly what HuggingFace will do on load. If this raises,
    # the pass did not achieve what it was for and the output must not be published.
    print("\n  verifying each tier resolves to a single schema:")
    ok = True
    for tier in args.tiers:
        root = args.out_dir / tier
        files = sorted(root.glob("**/*.parquet"))
        if not files:
            continue
        try:
            lf = pl.scan_parquet(files)
            schema = lf.collect_schema()
            rows = lf.select(pl.len()).collect().item()
            print(
                f"    {tier}: {len(files):,} files, {len(schema)} columns, {rows:,} rows -> UNIFIES"
            )
        except Exception as exc:  # noqa: BLE001 - the verdict is the point, not the traceback
            ok = False
            print(f"    {tier}: DOES NOT UNIFY -- {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0 if ok else 3


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the requested mode."""
    args = parse_args(argv)
    return do_inspect(args) if args.mode == "inspect" else do_run(args)


if __name__ == "__main__":
    sys.exit(main())
