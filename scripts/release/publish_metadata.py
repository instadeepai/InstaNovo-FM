# ruff: noqa: T201 - a CLI step: the printed record of what was sent is the point
r"""Upload the dataset card and the release manifests.

Separate from the tier upload because these are the files that make the data
*intelligible* rather than present, and they go last on purpose. Until the card is in
place the repository has no ``configs:`` block, so ``load_dataset(..., "hcfm_splits")``
cannot resolve however complete the parquet is; and a card that described configs before
their data existed would be worse than none.

Two things are uploaded:

* ``README.md`` -- the dataset card, whose front matter declares every config. This is
  the file that turns 45,000 parquet files into seven named, loadable datasets.
* ``manifests/`` -- currently ``empty_runs.csv``, the runs that yielded no PSMs at their
  tier's confidence threshold. An empty high-confidence run is a finding about that run,
  so it is recorded rather than silently dropped.

``upload_file`` rather than ``upload_large_folder``: these are small, they need to land
in one visible commit each, and the large-folder path deliberately gives no control over
the commit message.

    python scripts/release/publish_metadata.py \
        --card scripts/release/dataset_card.md --stage "$STAGE"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from instanovo_fm.utils.hf_token import resolve_hf_token
except ImportError:  # pragma: no cover - the slim runner image has no instanovo_fm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_token import resolve_hf_token  # type: ignore[no-redef]

DEFAULT_REPO = "InstaDeepAI/InstaNovo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--card", type=Path, required=True, help="dataset card to send as README.md")
    p.add_argument("--stage", type=Path, default=None, help="staging root holding manifests/")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--repo-type", default="dataset")
    p.add_argument(
        "--tag",
        default=None,
        help="create this tag after the upload, pinning the commit the upload produced",
    )
    p.add_argument("--tag-message", default=None, help="annotation for the tag")
    p.add_argument(
        "--move-tag",
        action="store_true",
        help="delete the tag first if it exists, so it can be repointed at the new commit",
    )
    p.add_argument("--dry-run", action="store_true", help="report what would be sent")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Upload the card and manifests, then report what the repo holds."""
    args = parse_args(argv)
    if not args.card.is_file():
        raise SystemExit(f"error: --card {args.card} is not a file")

    # Fail before uploading rather than publishing a card whose config block is broken:
    # a malformed front matter leaves every config unresolvable.
    import yaml

    text = args.card.read_text()
    if not text.startswith("---"):
        raise SystemExit("error: card has no YAML front matter, so it declares no configs")
    front = yaml.safe_load(text.split("---")[1])
    configs = front.get("configs") or []
    if not configs:
        raise SystemExit("error: card front matter declares no configs")
    print(f"  card declares {len(configs)} config(s):")
    for c in configs:
        where = c.get("data_dir") or c["data_files"][0]["path"]
        default = " (default)" if c.get("default") else ""
        print(f"    {c['config_name']}{default} -> {where}")

    sends: list[tuple[Path, str]] = [(args.card, "README.md")]
    if args.stage:
        manifests = sorted((args.stage / "manifests").glob("*")) if args.stage else []
        sends += [(m, f"manifests/{m.name}") for m in manifests if m.is_file()]

    print()
    for local, remote in sends:
        print(f"  {local}  ->  {remote}  ({local.stat().st_size:,} bytes)")
    if args.dry_run:
        print("\n  dry run: nothing sent")
        return 0

    token, var = resolve_hf_token()
    print(f"\n  token from {var}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for local, remote in sends:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Add {remote}",
        )
        print(f"  sent {remote}")

    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
    print(f"\n  repo now holds {len(files):,} files")
    missing = [r for _, r in sends if r not in files]
    for remote in (r for _, r in sends):
        print(f"    {remote}: {'present' if remote in files else 'MISSING'}")
    if missing:
        print(f"\n  refusing to tag: {missing} did not land")
        return 1

    if args.tag:
        # Tag last, and only once the card is in place. A tag taken before it would pin a
        # tree whose configs are undeclared, and a tag cannot be meaningfully repaired --
        # anyone who pinned it keeps the broken revision.
        info = api.repo_info(args.repo_id, repo_type=args.repo_type)
        sha = info.sha
        existing = {
            t.name: t.target_commit
            for t in api.list_repo_refs(args.repo_id, repo_type=args.repo_type).tags
        }
        if args.tag in existing:
            if not args.move_tag:
                print(
                    f"\n  {args.tag} already exists at {existing[args.tag]}; refusing to touch it."
                    f"\n  Pass --move-tag only while nobody can have pinned it -- moving a tag"
                    f"\n  someone has pinned changes what they get without their knowing."
                )
                return 1
            print(f"\n  {args.tag} exists at {existing[args.tag]}; deleting to repoint it")
            api.delete_tag(repo_id=args.repo_id, tag=args.tag, repo_type=args.repo_type)
        print(f"\n  tagging {args.tag} at {sha}")
        api.create_tag(
            repo_id=args.repo_id,
            tag=args.tag,
            tag_message=args.tag_message or f"Release {args.tag}",
            repo_type=args.repo_type,
            revision=sha,
        )
        refs = api.list_repo_refs(args.repo_id, repo_type=args.repo_type)
        tags = {t.name: t.target_commit for t in refs.tags}
        if args.tag not in tags:
            print(f"  TAG MISSING after creation: {sorted(tags)}")
            return 1
        print(f"  tag {args.tag} -> {tags[args.tag]}")
        print(f"\n  CITE THIS: revision={args.tag} (commit {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
