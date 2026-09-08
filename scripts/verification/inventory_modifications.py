# ruff: noqa: T201 - a CLI report: the printed inventory is the whole point
r"""Inventory every modification token in the ``sequence`` column of a corpus.

``list_in_sequence_modifications.py`` answers a narrow question: which ``[IN:<digits>]``
tokens occur. This answers the general one -- **every** bracketed token, with the
residue it attaches to -- which is worth having as a permanent reference rather than a
one-off check.

Why the general form matters:

* It confirms or refutes the claim that ``[IN:...]`` is glyco-only, by showing which
  residues those tokens attach to. The annotation pipeline assigns
  ``N[IN:3000]``--``N[IN:3172]`` to N-glycan compositions absent from UNIMOD, so every
  ``[IN:]`` token should sit on an N. Anything else came from a different source.
* It surfaces **unknown notation**. A token that is neither ``[UNIMOD:n]`` nor
  ``[IN:n]`` is either an unmapped mass delta or a format this pipeline does not
  understand, and either way a downstream user cannot resolve it.
* It is the modification vocabulary of the published corpus, which belongs in the
  release as ``manifests/modifications.csv``. Users ask what PTMs a dataset contains,
  and today the only answer is "read 46,367 files".

**Scope: only the six datasets intended for the HuggingFace release** --
``{hcfm,mcfm,lcfm}_splits`` and the three unsplit tier directories. ACFM is excluded
because it is not published, and so are the other stores on the mounts. The point is a
reference for what ships, not a survey of everything on disk.

Reads only ``sequence``, so it touches a small fraction of the corpus.

Output: one CSV row per (config, project, residue, token) with row counts, plus a
printed summary grouped by token kind.

Usage:
    python scripts/verification/inventory_modifications.py
    python scripts/verification/inventory_modifications.py --configs lcfm_by_project
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for annotations; polars is imported lazily below
    import polars as pl

# A residue letter if present (so N-glycans read as "N"), then a modification token.
# Both bracket styles: search engines emit [UNIMOD:35] and [+79.966] but also (ox),
# and the pipeline's own normaliser strips round and square alike -- so matching only
# square brackets would make a whole notation invisible.
# The regex engine has no look-behind, so the letter is part of the match.
TOKEN_RE = r"[A-Za-z]?(?:\[[^\]]*\]|\([^\)]*\))"
SPLIT_RE = re.compile(r"^([A-Za-z]?)([\[(].*[\])])$")

UNIMOD_RE = re.compile(r"^\[UNIMOD:\d+\]$", re.I)
IN_RE = re.compile(r"^\[IN:\d+\]$", re.I)

# config name -> directory on the mount. These six are the release; ACFM is not
# published and is deliberately absent.
RELEASE_CONFIGS: dict[str, str] = {
    "hcfm_splits": "hcfm_splits",
    "mcfm_splits": "mcfm_splits",
    "lcfm_splits": "lcfm_splits",
    "hcfm_by_project": "hcfm",
    "mcfm_by_project": "mcfm",
    "lcfm_by_project": "lcfm",
}


def classify(token: str) -> str:
    """Bucket a bracketed token by notation."""
    if UNIMOD_RE.match(token):
        return "unimod"
    if IN_RE.match(token):
        return "in"
    return "other"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mount",
        type=Path,
        required=True,
        help="root holding the tier directories (required; no default, since the path is "
        "deployment-specific)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(RELEASE_CONFIGS),
        choices=list(RELEASE_CONFIGS),
        help="release configs to inventory; the default is all six",
    )
    p.add_argument("--output-csv", type=Path, default=Path("modifications.csv"))
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--progress-every", type=int, default=2000)
    return p.parse_args(argv)


def _targets(mount: Path, configs: list[str]) -> list[tuple[str, Path]]:
    """Resolve the requested release configs to existing directories."""
    out = []
    for cfg in configs:
        root = mount / RELEASE_CONFIGS[cfg]
        if root.is_dir():
            out.append((cfg, root))
        else:
            print(f"  [warn] absent, skipping: {cfg} -> {root}")
    return out


def _enumerate(targets: list[tuple[str, Path]]) -> list[tuple[str, Path, Path]]:
    """List every non-empty file per config."""
    files: list[tuple[str, Path, Path]] = []
    for cfg, root in targets:
        found = sorted(f for f in root.rglob("*.parquet") if f.is_file() and f.stat().st_size)
        if not found:
            found = sorted(f for f in root.rglob("*") if f.is_file() and f.stat().st_size)
        print(f"    {cfg:18s} {len(found):>7,} non-empty files under {root}")
        files.extend((cfg, root, f) for f in found)
    return files


def _read_sequences(path: Path, seq_col: str) -> "pl.DataFrame":
    """Read just the sequence column. Raises KeyError if the column is absent."""
    import polars as pl

    lf = pl.scan_parquet(path)
    if seq_col not in lf.collect_schema().names():
        raise KeyError(seq_col)
    return lf.select(pl.col(seq_col).cast(pl.Utf8, strict=False).fill_null("")).collect()


def _tally_tokens(df: "pl.DataFrame", seq_col: str) -> tuple[int, dict[tuple[str, str], int]]:
    """Return (rows carrying at least one token, {(residue, token): occurrences})."""
    import polars as pl

    toks = df.select(pl.col(seq_col).str.extract_all(TOKEN_RE).alias("t"))
    non_empty = toks.filter(pl.col("t").list.len() > 0)
    # Filter before exploding: an empty list explodes to null today and to an empty
    # string under the polars 2.0 default, so neither behaviour is relied on.
    flat = non_empty.explode("t").drop_nulls("t").filter(pl.col("t") != "")
    out: dict[tuple[str, str], int] = {}
    if flat.height:
        vc = flat["t"].value_counts()
        for tok, n in zip(vc["t"].to_list(), vc["count"].to_list(), strict=True):
            m = SPLIT_RE.match(tok)
            residue, bracket = (m.group(1) or "-", m.group(2)) if m else ("?", tok)
            key = (residue, bracket)
            out[key] = out.get(key, 0) + int(n)
    return int(non_empty.height), out


def _scan(files: list[tuple[str, Path, Path]], seq_col: str, every: int) -> dict:
    """Count modification tokens per (config, project, residue, token)."""
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    per_config: dict[str, int] = {}
    skipped: list[str] = []
    total = 0
    with_mod = 0

    for i, (cfg, root, f) in enumerate(files, 1):
        rel = f.relative_to(root)
        project = rel.parts[0] if len(rel.parts) > 1 else "(flat)"
        try:
            df = _read_sequences(f, seq_col)
        except KeyError:
            skipped.append(f"{cfg}/{rel}: no {seq_col} column")
            continue
        except Exception as exc:  # noqa: BLE001 - a bad file is a finding
            skipped.append(f"{cfg}/{rel}: {type(exc).__name__}: {exc}"[:200])
            continue

        total += df.height
        per_config[cfg] = per_config.get(cfg, 0) + df.height
        n_with, tallies = _tally_tokens(df, seq_col)
        with_mod += n_with
        for (residue, bracket), n in tallies.items():
            counts[(cfg, project, residue, bracket)] += n
        if i % every == 0:
            print(f"    {i:,}/{len(files):,} rows={total:,}", flush=True)

    return {
        "counts": counts,
        "per_config": per_config,
        "skipped": skipped,
        "rows": total,
        "rows_with_token": with_mod,
    }


def _report_in_tokens(tok_rows: dict[str, int], tok_res: dict[str, set[str]]) -> list[str]:
    """Print the [IN:*] breakdown and return the residues those tokens attach to."""
    in_toks = sorted((t for t in tok_rows if classify(t) == "in"), key=lambda t: -tok_rows[t])
    print("\n  [IN:*] tokens — which residues do they attach to?")
    if not in_toks:
        print("    none present")
        return []
    residues = sorted({r for t in in_toks for r in tok_res[t]})
    verdict = "N only — glyco claim holds" if residues == ["N"] else "NOT N-only — investigate"
    print(f"    residues seen: {residues}   ({verdict})")
    ids = [int(m.group(1)) for t in in_toks if (m := re.search(r"IN:(\d+)", t))]
    if ids:
        outside = sorted(i for i in ids if not 3000 <= i <= 3172)
        print(
            f"    id range: {min(ids)}-{max(ids)}; outside the mapped 3000-3172 block: {outside or 'none'}"
        )
    for t in in_toks[:10]:
        print(f"      {''.join(sorted(tok_res[t]))}{t:14s} {tok_rows[t]:>14,}")
    if len(in_toks) > 10:
        print(f"      ... and {len(in_toks) - 10} more")
    return residues


def _report(res: dict, configs: list[str]) -> tuple[dict, list[str], list[str]]:
    """Print the inventory summary. Returns (by_kind, in_residues, unrecognised)."""
    counts = res["counts"]
    by_kind: dict[str, int] = defaultdict(int)
    tok_rows: dict[str, int] = defaultdict(int)
    tok_res: dict[str, set[str]] = defaultdict(set)
    for (_, _, residue, tok), n in counts.items():
        by_kind[classify(tok)] += n
        tok_rows[tok] += n
        tok_res[tok].add(residue)

    print("\n  rows per config:")
    for cfg in configs:
        if cfg in res["per_config"]:
            print(f"    {cfg:18s} {res['per_config'][cfg]:>16,}")

    total = res["rows"]
    pct = 100.0 * res["rows_with_token"] / total if total else 0.0
    print(f"\n  rows scanned          : {total:,}")
    print(f"  rows with >=1 token   : {res['rows_with_token']:,} ({pct:.2f}%)")
    print(f"  distinct tokens       : {len(tok_rows):,}")
    for kind in ("unimod", "in", "other"):
        n_tok = sum(1 for t in tok_rows if classify(t) == kind)
        print(f"    {kind:7s} {n_tok:>5,} distinct, {by_kind[kind]:>16,} token occurrences")

    residues = _report_in_tokens(tok_rows, tok_res)

    other = sorted((t for t in tok_rows if classify(t) == "other"), key=lambda t: -tok_rows[t])
    print(f"\n  UNRECOGNISED notation: {len(other):,} distinct token(s)")
    for t in other[:15]:
        print(f"      {''.join(sorted(tok_res[t]))}{t:24s} {tok_rows[t]:>14,}")
    if len(other) > 15:
        print(f"      ... and {len(other) - 15} more")

    if res["skipped"]:
        print(f"\n  SKIPPED {len(res['skipped'])} file(s) — counts are incomplete:")
        for x in res["skipped"][:5]:
            print(f"      {x}")

    return dict(by_kind), residues, other


def main(argv: list[str] | None = None) -> int:
    """Inventory modification tokens across the release configs."""
    args = parse_args(argv)
    if not args.mount.is_dir():
        raise SystemExit(f"error: {args.mount} is not a directory")

    targets = _targets(args.mount, args.configs)
    print(f"  inventorying {len(targets)} release config(s); ACFM is excluded by design")
    files = _enumerate(targets)
    print(f"  total files: {len(files):,}")

    res = _scan(files, args.sequence_col, args.progress_every)
    by_kind, residues, other = _report(res, args.configs)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "project", "residue", "token", "kind", "rows"])
        for (cfg, project, residue, tok), n in sorted(res["counts"].items()):
            w.writerow([cfg, project, residue, tok, classify(tok), n])
    print(f"\n  wrote {args.output_csv} ({len(res['counts']):,} rows)")

    tokens = {t for (_, _, _, t) in res["counts"]}
    summary = {
        "mount": str(args.mount),
        "configs": args.configs,
        "rows_per_config": res["per_config"],
        "rows": res["rows"],
        "rows_with_token": res["rows_with_token"],
        "distinct_tokens": len(tokens),
        "by_kind": {
            k: {
                "distinct": sum(1 for t in tokens if classify(t) == k),
                "occurrences": by_kind.get(k, 0),
            }
            for k in ("unimod", "in", "other")
        },
        "in_residues": residues,
        "unrecognised": other[:50],
        "skipped": res["skipped"],
    }
    summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}")
    print("\n  --- summary ---")
    for line in json.dumps(summary, indent=2).splitlines():
        print(f"  {line}")
    return 1 if res["skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
