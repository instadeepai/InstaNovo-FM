# ruff: noqa: T201 - a CLI step: the printed upload manifest and progress are the
# output, and on a batch pod the log is the only record that outlives the job.
"""Upload the labelled confidence tiers to the HuggingFace dataset repo.

Each tier exists on disk in two forms, and **both are uploaded**, because the
manuscript relies on each for a different claim:

``<tier>_splits``
    Quality-filtered, shuffled ``train``/``test``/``valid`` parquet. This is what
    the model consumed -- the "training and evaluation data" of Data
    Availability. Five filters: retention time <= 10800 s, lower isolation offset
    <= 300 Da, precursor charge 0-7 inclusive (0 is kept), precursor m/z <= 2000, and
    no modification annotation unresolvable to a UNIMOD identifier (the `[IN:<digits>]`
    namespace; mostly N-glycans, but not exclusively). Nulls pass every numeric
    condition.

``by_project/<tier>``
    The tier before filtering and splitting, laid out one directory per accession.
    Needed because the
    filtering is lossy: rows the quality gates drop are not recoverable from the
    splits, so without this the peptide registry cannot be used to re-derive or
    extend the partitions.

Representation sits above tier in the repo, deliberately::

    splits/lcfm/      splits/mcfm/      splits/hcfm/       filtered, shuffled, disjoint
    by_project/lcfm/  by_project/mcfm/  by_project/hcfm/   complete tier, unfiltered

With tier on top instead, the one-level glob ``lcfm/*`` would quietly fetch both
representations of overlapping rows -- the easiest mistake to make and the most
expensive. This way no single-level glob can cross the boundary.

Paths here are identity mappings: the staging tree built by ``stage_release.py``
already mirrors the repository, so nothing is translated during upload and the tree
can be inspected beforehand exactly as it will be published.

The token is read from ``INSTANOVO_FM_HF_TOKEN``, falling back to
``INSTANOVO_HF_TOKEN``. The name actually used is printed, so a token set under the
wrong name is visible rather than surfacing later as a 404.

``--source-root`` is required and has no default: the path is deployment-specific
and hard-coding one would bake a stale location into a public repository.

ACFM is not handled. As Data Availability states, the unlabelled tier is
approximately 5.7 TB and is not redistributed; it comprises every MS/MS scan from
the same raw files, so it is reconstructible from the accessions with the
conversion pipeline deposited at Figshare. ``--tier acfm`` is rejected rather
than silently attempted.

Usage::

    # dry run first -- lists what would be sent, uploads nothing
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier hcfm --dry-run

    # then for real, smallest tier first so a problem surfaces cheaply
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier hcfm
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier mcfm
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier lcfm

    # one form only, if uploading them separately
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier lcfm --flavour splits

    # confirm without sending anything
    python scripts/release/hf_upload.py --source-root "$ROOT" --tier lcfm --verify-only

Uploads are resumable: re-running the same command skips files already present at
a matching size.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # huggingface_hub is imported late, after the Xet env is set
    from huggingface_hub import HfApi

try:
    from instanovo_fm.utils.hf_token import resolve_hf_token
except ImportError:  # pragma: no cover - the slim runner image has no instanovo_fm
    # The upload runs on a minimal image that carries the scripts but not the package,
    # so the resolver is vendored beside them. One implementation, two locations to
    # import it from; the vendored copy is kept identical.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_token import resolve_hf_token  # type: ignore[no-redef]

LABELLED_TIERS = ("lcfm", "mcfm", "hcfm")
DEFAULT_REPO = "InstaDeepAI/InstaNovo"

# flavour -> (path under the staging root, prefix in the repo); identical by design
FLAVOURS: dict[str, tuple[str, str]] = {
    "splits": ("splits/{tier}", "splits/{tier}"),
    "by_project": ("by_project/{tier}", "by_project/{tier}"),
}


@dataclass(frozen=True)
class LocalTier:
    """One flavour of one tier, resolved on disk."""

    tier: str
    flavour: str
    path: Path
    prefix: str
    files: tuple[Path, ...]

    @property
    def total_bytes(self) -> int:
        """Total size on disk of every file in this unit."""
        return sum(f.stat().st_size for f in self.files)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="directory holding the tier subdirectories (required; no default)",
    )
    p.add_argument("--tier", required=True, help=f"one of {', '.join(LABELLED_TIERS)}")
    p.add_argument(
        "--flavour",
        choices=["both", *FLAVOURS],
        default="both",
        help="'both' (default) uploads the split partitions and the per-project tier "
        "into separate top-level folders; pass one name to upload only that form",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument("--dry-run", action="store_true", help="report what would be sent, send nothing")
    p.add_argument(
        "--verify-only", action="store_true", help="compare local and remote, upload nothing"
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="parallel upload workers. Left unset, huggingface_hub uses "
        "max(cpu_count // 2, 1), so a 2-CPU container gets ONE worker and the "
        "upload is effectively serial. Default here is cpu_count, minimum 8.",
    )
    p.add_argument(
        "--no-high-performance",
        action="store_true",
        help="do not set HF_XET_HIGH_PERFORMANCE (set by default; the target repos "
        "are Xet-backed, and this is the supported high-throughput path)",
    )
    p.add_argument("--allow-acfm", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def human(n: float) -> str:
    """Format a byte count in the largest unit that keeps it readable."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def resolve(root: Path, tier: str, flavours: list[str]) -> list[LocalTier]:
    """Resolve the requested flavours on disk, erroring on any that is absent."""
    if not root.is_dir():
        raise SystemExit(f"error: --source-root {root} is not a directory")

    resolved, missing = [], []
    for flavour in flavours:
        dirname, prefix = FLAVOURS[flavour]
        path = root / dirname.format(tier=tier)
        if not path.is_dir():
            missing.append(f"{flavour} -> {path}")
            continue
        files = tuple(sorted(f for f in path.rglob("*") if f.is_file()))
        if not files:
            missing.append(f"{flavour} -> {path} (exists but empty)")
            continue
        resolved.append(
            LocalTier(
                tier=tier,
                flavour=flavour,
                path=path,
                prefix=prefix.format(tier=tier),
                files=files,
            )
        )

    if missing:
        raise SystemExit(
            "error: requested flavour(s) not found on disk:\n  " + "\n  ".join(missing)
        )
    return resolved


def remote_sizes(
    api: HfApi, repo_id: str, repo_type: str, prefix: str, token: str
) -> dict[str, int]:
    """Map repo-relative path -> size for everything already under ``prefix``."""
    sizes: dict[str, int] = {}
    try:
        tree = api.list_repo_tree(
            repo_id, repo_type=repo_type, path_in_repo=prefix, recursive=True, token=token
        )
    except Exception:  # noqa: BLE001 - an absent prefix simply means nothing uploaded yet
        return sizes
    for info in tree:
        size = getattr(info, "size", None)
        if size is not None:  # directories carry none
            sizes[info.path] = size
    return sizes


def verify(local: LocalTier, remote: dict[str, int]) -> tuple[list[str], list[str]]:
    """Return (missing, size_mismatch) as repo-relative paths."""
    missing, mismatch = [], []
    for f in local.files:
        rel = f"{local.prefix}/{f.relative_to(local.path).as_posix()}"
        size = f.stat().st_size
        if rel not in remote:
            missing.append(rel)
        elif remote[rel] != size:
            mismatch.append(f"{rel} (local {size} vs remote {remote[rel]})")
    return missing, mismatch


def upload_one(
    api: HfApi, args: argparse.Namespace, local: LocalTier, token: str, workers: int
) -> int:
    """Upload and verify a single flavour. Returns 0 on success."""
    print("-" * 78)
    print(local.prefix)
    print(f"  source : {local.path}")
    print(f"  target : {args.repo_id}:{local.prefix}")
    print(f"  files  : {len(local.files):,}   size: {human(local.total_bytes)}")

    if args.dry_run:
        for f in local.files[:5]:
            rel = f.relative_to(local.path).as_posix()
            print(f"    would send {local.prefix}/{rel} ({human(f.stat().st_size)})")
        if len(local.files) > 5:
            print(f"    ... and {len(local.files) - 5:,} more")
        return 0

    if not args.verify_only:
        # upload_large_folder is resumable and multi-threaded, and is the
        # supported path for folders of this size.
        uploader = getattr(api, "upload_large_folder", None)
        if uploader is not None:
            # upload_large_folder has no path_in_repo: it mirrors the local tree at the
            # repo root, and takes no token either. So point it at the staging root --
            # which is built to mirror the repository exactly, for this reason -- and
            # select one unit with allow_patterns rather than uploading a subfolder.
            uploader(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(args.source_root),
                allow_patterns=[f"{local.prefix}/**"],
                num_workers=workers,
            )
        else:
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(local.path),
                path_in_repo=local.prefix,
                token=token,
                commit_message=(
                    f"Add {local.prefix} "
                    f"({len(local.files):,} files, {human(local.total_bytes)})"
                ),
            )

    remote = remote_sizes(api, args.repo_id, args.repo_type, local.prefix, token)
    missing, mismatch = verify(local, remote)
    print(
        f"  verified: remote {len(remote):,} files, "
        f"missing {len(missing):,}, mismatch {len(mismatch):,}"
    )
    for m in missing[:5]:
        print(f"    MISSING  {m}")
    for m in mismatch[:5]:
        print(f"    MISMATCH {m}")
    if missing or mismatch:
        print(f"  INCOMPLETE: {local.prefix} -- re-run to resume")
        return 1
    print(f"  COMPLETE: {local.prefix}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Resolve the requested tier and flavours, then upload or report them."""
    args = parse_args(argv)
    tier = args.tier.lower()

    if tier == "acfm" and not args.allow_acfm:
        raise SystemExit(
            "error: ACFM is not uploaded. Data Availability states it is approximately "
            "5.7 TB and not redistributed, being reconstructible from the accessions "
            "with the conversion pipeline deposited at Figshare."
        )
    if tier not in LABELLED_TIERS:
        raise SystemExit(f"error: --tier must be one of {', '.join(LABELLED_TIERS)}, got '{tier}'")

    try:
        token, token_var = resolve_hf_token()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc

    # HF_XET_HIGH_PERFORMANCE is read when huggingface_hub is imported, so it has to
    # be set before that import below. hf_transfer is deprecated and superseded by
    # this; huggingface_hub warns if you set the old variable.
    if not args.no_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    cores = os.cpu_count() or 1
    workers = args.num_workers if args.num_workers else max(cores, 8)

    flavours = list(FLAVOURS) if args.flavour == "both" else [args.flavour]
    locals_ = resolve(args.source_root, tier, flavours)

    print("=" * 78)
    print(f"HuggingFace tier upload  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  repo    : {args.repo_id} (repo_type={args.repo_type})")
    print(f"  token   : {token_var}")
    print(
        f"  workers : {workers} (host has {cores} core(s); "
        f"huggingface_hub would default to {max(cores // 2, 1)})"
    )
    print(
        f"  xet     : high-performance "
        f"{'ON' if os.environ.get('HF_XET_HIGH_PERFORMANCE') else 'off'}"
    )
    if cores < 8:
        print(
            f"  WARNING : only {cores} core(s) visible. Upload throughput scales with "
            f"workers, and each worker also needs CPU for sha256 and chunking. Give "
            f"the job more CPUs."
        )
    print(f"  tier    : {tier}")
    print(f"  flavours: {', '.join(f'{t.flavour} -> {t.prefix}' for t in locals_)}")
    total = sum(t.total_bytes for t in locals_)
    print(f"  total   : {sum(len(t.files) for t in locals_):,} files, {human(total)}")
    print("=" * 78)

    from huggingface_hub import HfApi

    # The token goes on the client, not on the call: upload_large_folder takes no
    # token argument.
    api = HfApi(token=token)
    failures = sum(upload_one(api, args, local, token, workers) for local in locals_)

    print("=" * 78)
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0
    if failures:
        print(f"RESULT: {failures} of {len(locals_)} incomplete for {tier}. Re-run to resume.")
        return 1
    print(f"RESULT: {tier} complete -- {', '.join(t.prefix for t in locals_)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
