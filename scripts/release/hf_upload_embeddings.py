#!/usr/bin/env python
"""Publish the embedding shards to a HuggingFace dataset repo.

Separate from ``hf_upload.py``, which sends the spectra corpus to
``InstaDeepAI/InstaNovo`` in tiers and flavours. This sends a different artefact -- derived
embeddings -- to a different repo, under a different licence, and shares none of that
script's tier logic. Bolting a mode onto it would have meant a flag that changes what
every other flag means.

The order of operations mirrors ``scripts/release/README.md``, because the failure modes
are the same and worth separating before any data moves:

    export INSTANOVO_FM_HF_TOKEN=...          # write access to the target repo

    # 1. What is on disk, what it weighs, and does the token actually permit the write.
    python scripts/release/hf_upload_embeddings.py --source-root <dir> --check

    # 2. What would be sent, byte for byte. Sends nothing, creates nothing.
    python scripts/release/hf_upload_embeddings.py --source-root <dir> --dry-run

    # 3. Smallest config first, so a configuration problem surfaces cheaply.
    python scripts/release/hf_upload_embeddings.py --source-root <dir> --config 100k
    python scripts/release/hf_upload_embeddings.py --source-root <dir> --config 1M

    # 4. Compare local against remote without sending anything.
    python scripts/release/hf_upload_embeddings.py --source-root <dir> --verify-only

``--source-root`` is required and has no default: the path is deployment-specific and
hard-coding one would bake a stale location into a public repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPO = "InstaDeepAI/InstaNovo-FM-embeddings"
CARD = Path(__file__).with_name("embeddings_card.md")
TOKEN_VARS = ("INSTANOVO_FM_HF_TOKEN", "INSTANOVO_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _log(msg: str = "") -> None:
    print(msg, flush=True)  # noqa: T201


def find_token() -> tuple[str | None, str]:
    """The token and where it came from, so a permission failure is explicable.

    Falls back to the cached ``huggingface-cli login`` rather than treating an unset
    environment variable as no credential at all -- the whoami check passes on the cached
    token, so refusing to upload with it reported "no token" about a token that works.
    """
    for name in TOKEN_VARS:
        value = os.environ.get(name)
        if value:
            return value, name
    from huggingface_hub import get_token

    cached = get_token()
    return cached, "the cached huggingface-cli login" if cached else "nowhere"


def discover(root: Path, only: str | None) -> dict[str, dict[str, Any]]:
    """Configs on disk, from the summaries `prepare_embeddings.py` wrote beside them."""
    configs: dict[str, dict[str, Any]] = {}
    for summary in sorted(root.glob("*.summary.json")):
        meta = json.loads(summary.read_text())
        name = meta["config"]
        if only and name != only:
            continue
        directory = root / name
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise SystemExit(f"{directory} has no parquet shards; run prepare_embeddings.py first")
        declared = {s["file"] for s in meta["shards"]}
        found = {f.name for f in files}
        if declared != found:
            raise SystemExit(
                f"{name}: the summary names {sorted(declared - found)} which are absent, "
                f"and {sorted(found - declared)} are present but undeclared -- the "
                "directory and the summary disagree, so one of them is stale"
            )
        configs[name] = {"meta": meta, "files": files,
                         "bytes": sum(f.stat().st_size for f in files)}
    if only and not configs:
        raise SystemExit(f"no config named {only!r} under {root}")
    if not configs:
        raise SystemExit(f"no *.summary.json under {root}; run prepare_embeddings.py first")
    return configs


def describe(configs: dict[str, dict[str, Any]]) -> None:
    """Say what is on disk before anything is sent."""
    total = 0
    for name, c in configs.items():
        meta = c["meta"]
        _log(f"  {name}: {meta['rows']:,} rows x {meta['embedding_dim']}-d "
             f"{meta['embedding_pooling']}, {len(c['files'])} shard(s), "
             f"{c['bytes'] / 2**20:.0f} MiB")
        _log(f"      metadata fields  : {len(meta['metadata_fields'])} "
             f"(skipped {len(meta['skipped_fields'])}: {', '.join(meta['skipped_fields'])})")
        _log(f"      coordinates      : {', '.join(meta['coordinate_columns']) or '(none)'}")
        total += c["bytes"]
    _log(f"  total to send: {total / 2**20:.0f} MiB")


def remote_sizes(api: Any, repo_id: str, token: str | None) -> dict[str, int]:
    """Sizes already in the repo, empty if it does not exist yet."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.dataset_info(repo_id, files_metadata=True, token=token)
    except RepositoryNotFoundError:
        return {}
    return {s.rfilename: (s.size or 0) for s in (info.siblings or [])}


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0915 -- a linear runbook
    """Check, dry-run, upload or verify."""
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, required=True,
                    help="directory holding <config>/ and <config>.summary.json")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--config", help="upload one config; default is all of them")
    ap.add_argument("--check", action="store_true",
                    help="report the environment and what is on disk; send nothing")
    ap.add_argument("--dry-run", action="store_true", help="report what would be sent; send nothing")
    ap.add_argument("--verify-only", action="store_true",
                    help="compare local against remote; send nothing")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (it is public by default)")
    args = ap.parse_args(argv)

    configs = discover(args.source_root, args.config)
    token, token_source = find_token()
    api = HfApi()

    _log(f"repo    : {args.repo_id} (dataset)")
    _log(f"token   : {token_source}")
    _log(f"card    : {CARD.name} ({CARD.stat().st_size:,} bytes)")
    _log("on disk :")
    describe(configs)

    # Whether the token can actually write here, reported before any transfer rather
    # than discovered halfway through one.
    try:
        me = api.whoami(token=token)
        org = args.repo_id.split("/")[0]
        roles = {o.get("name"): o.get("roleInOrg") for o in me.get("orgs", [])}
        role = roles.get(org) or ("owner" if me.get("name") == org else None)
        _log(f"identity: {me.get('name')} | role in {org}: {role or 'NONE'}")
        if role not in {"write", "admin", "owner"}:
            _log(f"  the token cannot write to {org}; upload would fail")
            if not (args.check or args.dry_run):
                return 1
    except Exception as e:  # noqa: BLE001 -- report, do not raise past the runbook
        _log(f"identity: could not be established ({type(e).__name__}: {e})")
        if not (args.check or args.dry_run):
            return 1

    remote = remote_sizes(api, args.repo_id, token)
    _log(f"remote  : {'absent, would be created' if not remote else f'{len(remote)} files present'}")

    if args.verify_only:
        _log()
        ok = True
        for name, c in configs.items():
            for f in c["files"]:
                key = f"{name}/{f.name}"
                have, want = remote.get(key), f.stat().st_size
                if have is None:
                    _log(f"  MISSING  {key}")
                    ok = False
                elif have != want:
                    _log(f"  SIZE     {key}: remote {have:,} != local {want:,}")
                    ok = False
        _log("  every shard present at the right size" if ok else "  the repo does not match this build")
        return 0 if ok else 1

    if args.check or args.dry_run:
        _log()
        _log("would send:")
        for name, c in configs.items():
            for f in c["files"]:
                key = f"{name}/{f.name}"
                state = "new" if key not in remote else (
                    "unchanged" if remote[key] == f.stat().st_size else "replaced")
                _log(f"  {state:9s} {key}  ({f.stat().st_size / 2**20:.0f} MiB)")
        _log(f"  {'new' if 'README.md' not in remote else 'replaced':9s} README.md  (the dataset card)")
        _log()
        _log("nothing was sent and no repo was created")
        return 0

    if token is None:
        _log("no token; set one of " + ", ".join(TOKEN_VARS))
        return 1

    api.create_repo(args.repo_id, repo_type="dataset", private=args.private,
                    exist_ok=True, token=token)
    _log(f"repo ready: {args.repo_id}")

    # The card first: a repo that exists without one is a repo whose terms are unstated.
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset", token=token,
                    commit_message="Add the dataset card")
    _log("uploaded README.md")

    for name, c in configs.items():
        _log(f"uploading {name}: {len(c['files'])} shard(s), {c['bytes'] / 2**20:.0f} MiB")
        api.upload_folder(
            folder_path=str(args.source_root / name), path_in_repo=name,
            repo_id=args.repo_id, repo_type="dataset", token=token,
            allow_patterns=["*.parquet"],
            commit_message=f"Add the {name} embeddings ({c['meta']['rows']:,} spectra)",
        )
        _log(f"  {name} sent")

    _log()
    _log(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
