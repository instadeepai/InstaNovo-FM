# ruff: noqa: T201 - a CLI step: the printed verdict per config is the point
r"""Load the published dataset the way a reader will, and check it against what we measured.

A config that has never been loaded end to end may not be loadable at all, and that
cannot be fixed inside a tag once one exists. The three per-project configs are the ones
this really tests: before the preparation pass they could not load, because a config is a
single unified scan and their files disagreed on three dtypes.

Two depths, chosen by cost:

* **Full load** of the smallest tier. ``load_dataset`` with no streaming, exactly as a
  reader would, then an exact row count against the number measured on the mount. This
  is the only check that proves the whole path -- config resolution, split detection,
  schema unification and decoding.
* **Resolution only** for the larger tiers. Streaming resolves the config and its splits
  and reads the first row without fetching the rest, which is enough to catch a wrong
  path or a split that does not exist. Downloading lcfm to count its rows would move
  ~900 GB to learn what the footers already told us.

The dataset viewer's size endpoint is queried too, but its numbers are advisory: it
reports ``estimated_num_rows`` precisely because at this scale it only partially
converts a dataset, and it returns nothing at all for a config that fails to load.

``HF_HOME`` should point somewhere with room -- a full load of one tier caches tens of
gigabytes, and the pod's own filesystem is not the place for it.

    HF_HOME=/mnt/.../hf_cache python scripts/release/verify_published.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

try:
    from instanovo_fm.utils.hf_token import resolve_hf_token
except ImportError:  # pragma: no cover - the slim runner image has no instanovo_fm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_token import resolve_hf_token  # type: ignore[no-redef]

DEFAULT_REPO = "InstaDeepAI/InstaNovo"

# Row counts measured on the mount, not copied from the manuscript. splits totals come
# from the filter audit; by_project totals from the preparation pass's own verification.
EXPECTED_ROWS = {
    "hcfm_splits": 3_670_113,
    "mcfm_splits": 18_255_265,
    "lcfm_splits": 181_777_591,
    "hcfm_by_project": 3_684_448,
    "mcfm_by_project": 18_422_236,
    "lcfm_by_project": 184_607_213,
}

# Fully loaded and counted; the rest are resolved only.
FULL = ("hcfm_splits", "hcfm_by_project")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--full", nargs="*", default=list(FULL), help="configs to load and count")
    p.add_argument(
        "--configs",
        nargs="*",
        default=list(EXPECTED_ROWS),
        help="configs to check at all (default: every published config)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="retries per file download; a single transient fetch should not fail a tier",
    )
    p.add_argument("--out", type=Path, default=None, help="write the verdicts as JSON")
    return p.parse_args(argv)


def viewer_size(repo_id: str, token: str) -> dict:
    """Query the dataset viewer's size endpoint; advisory only."""
    url = f"https://datasets-server.huggingface.co/size?dataset={repo_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            payload: dict = json.load(fh)
        return payload
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    """Verify every published config, and count rows for the affordable ones."""
    args = parse_args(argv)
    token, var = resolve_hf_token()
    print(f"  token from {var}")
    # Set before datasets is imported. A private repo answers an unauthenticated file
    # request with 401, which huggingface_hub reports as RepositoryNotFoundError and
    # datasets rewraps as FileNotFoundError -- so a missing token looks exactly like
    # missing data. Belt and braces: explicit token arguments below, plus the ambient
    # variable for any path that resolves credentials itself.
    os.environ["HF_TOKEN"] = token
    print(f"  HF_HOME  = {os.environ.get('HF_HOME', '(default)')}")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo_files = set(HfApi().list_repo_files(args.repo_id, repo_type="dataset", token=token))
    print(f"  repo holds {len(repo_files):,} files")

    results: dict[str, dict] = {}
    failures = 0

    for config in args.configs:
        expected = EXPECTED_ROWS[config]
        print(f"\n--- {config} (expect {expected:,} rows) ---", flush=True)
        entry: dict = {"expected_rows": expected}
        try:
            if config in args.full:
                from datasets import DownloadConfig

                ds = load_dataset(
                    args.repo_id,
                    config,
                    token=token,
                    download_config=DownloadConfig(token=token, max_retries=args.max_retries),
                )
                counts = {split: ds[split].num_rows for split in ds}
                total = sum(counts.values())
                entry |= {"mode": "full", "splits": counts, "rows": total}
                print(f"  splits: {counts}")
                print(f"  columns: {len(ds[next(iter(ds))].column_names)}")
                if total == expected:
                    print(f"  ROWS MATCH: {total:,}")
                    entry["verdict"] = "PASS"
                else:
                    print(f"  ROW MISMATCH: got {total:,}, expected {expected:,}")
                    entry["verdict"] = "FAIL"
                    failures += 1
            else:
                ds = load_dataset(args.repo_id, config, token=token, streaming=True)
                splits = list(ds)
                first = next(iter(ds[splits[0]]))
                entry |= {
                    "mode": "resolved",
                    "splits": splits,
                    "columns": len(first),
                    "verdict": "RESOLVES",
                }
                print(f"  splits: {splits}")
                print(f"  first row has {len(first)} columns -> RESOLVES")
        except Exception as exc:  # noqa: BLE001 - the verdict per config is what matters
            # The message alone said only that a file could not be located, naming
            # neither the file nor the HTTP status. The chain and traceback carry both.
            chain, cur = [], exc
            while cur is not None:
                chain.append(f"{type(cur).__name__}: {cur}")
                cur = cur.__cause__ or cur.__context__
                if len(chain) > 6:
                    break
            entry |= {"verdict": "ERROR", "error": chain[0], "chain": chain}
            print(f"  ERROR: {chain[0]}")
            for i, link in enumerate(chain[1:], 1):
                print(f"    caused by [{i}]: {link}")
            print("  traceback:")
            for line in traceback.format_exc().splitlines():
                print(f"    {line}")
            failures += 1
        results[config] = entry

    print("\n--- dataset viewer (advisory) ---")
    size = viewer_size(args.repo_id, token)
    if "error" in size:
        print(f"  unavailable: {size['error']}")
    else:
        for cfg in size.get("size", {}).get("configs", []):
            print(
                f"  {cfg.get('config')}: {cfg.get('num_rows')} rows, "
                f"{cfg.get('num_bytes_original_files')} bytes"
            )
    results["_viewer"] = size

    print("\n" + "=" * 70)
    for config in args.configs:
        print(f"  {config:<18} {results[config].get('verdict')}")
    print("=" * 70)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"  wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
