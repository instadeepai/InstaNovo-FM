# ruff: noqa: T201 - a CLI probe: the printed answer per check is the point
"""Check HuggingFace connectivity, credentials and write access before uploading.

Run this first. It answers, separately, the questions a failed upload would
otherwise conflate: is the token present, can this host reach huggingface.co,
does the token authenticate, can we read the repo, and can we *write* to it.

The write check commits a small file under ``.hf-write-probe/`` and deletes it
again, so nothing is left behind. The token value is never printed -- only its
length and a truncated digest, so two runs can be compared without exposing it.

Usage::

    python scripts/release/hf_probe.py --repo-id InstaDeepAI/InstaNovo
    python scripts/release/hf_probe.py --source-root /path/to/tiers   # also check the source

Exit status is 0 only if every check needed for uploading passed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from instanovo_fm.utils.hf_token import resolve_hf_token

TIERS = ("lcfm", "mcfm", "hcfm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-id", default="InstaDeepAI/InstaNovo", help="target dataset repo")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="optional: also report the file count and size of both flavours of each tier",
    )
    p.add_argument("--skip-write", action="store_true", help="check read access only")
    return p.parse_args(argv)


class Probe:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn, critical: bool = True) -> None:
        try:
            detail = fn() or ""
            self.results.append((name, True, str(detail)))
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:  # noqa: BLE001 - report every failure, never abort early
            self.results.append((name, not critical, f"{type(exc).__name__}: {exc}"))
            print(f"[{'FAIL' if critical else 'WARN'}] {name}: {type(exc).__name__}: {exc}")

    def summary(self) -> int:
        print("=" * 78)
        failed = [n for n, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        print("=" * 78)
        if failed:
            print(f"RESULT: FAILED -- {len(failed)} check(s): {failed}")
            return 1
        print("RESULT: all checks passed")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    probe = Probe()
    print("=" * 78)
    print(f"HuggingFace probe  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"target: {args.repo_id} (repo_type={args.repo_type})")
    print("=" * 78)

    def token_present() -> str:
        try:
            tok, name = resolve_hf_token()
        except RuntimeError as exc:
            visible = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper())
            raise RuntimeError(
                f"{exc}. HF-ish variables visible here: {visible or 'none'}"
            ) from exc
        # Name the variable actually used: with two accepted names, "the token is
        # wrong" and "the token I set is not the one being read" look identical.
        return (
            f"from {name}: {len(tok)} chars, "
            f"sha256:{hashlib.sha256(tok.encode()).hexdigest()[:12]}"
        )

    def dns() -> str:
        return f"huggingface.co -> {socket.gethostbyname('huggingface.co')}"

    def egress() -> str:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2", headers={"User-Agent": "instanovo-fm-probe"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return f"reachable, unauthenticated status {resp.status}"
        except urllib.error.HTTPError as exc:
            # 401 is the expected unauthenticated answer and still proves egress
            return f"reachable, unauthenticated status {exc.code} (401 expected)"

    def whoami() -> str:
        from huggingface_hub import HfApi

        info = HfApi().whoami(token=resolve_hf_token()[0])
        orgs = [o.get("name") for o in info.get("orgs", [])]
        auth = info.get("auth", {})
        role = auth.get("accessToken", {}).get("role") or auth.get("type")
        return f"name={info.get('name')} type={info.get('type')} orgs={orgs} role={role}"

    def readable() -> str:
        from huggingface_hub import HfApi

        api = HfApi()
        tok = resolve_hf_token()[0]
        info = api.repo_info(args.repo_id, repo_type=args.repo_type, token=tok)
        files = api.list_repo_files(args.repo_id, repo_type=args.repo_type, token=tok)
        return f"private={info.private} sha={(info.sha or '')[:8]} files={len(files)}"

    def writable() -> str:
        from huggingface_hub import HfApi

        api = HfApi()
        tok = resolve_hf_token()[0]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = os.environ.get("AICHOR_EXPERIMENT_ID", "local")
        path_in_repo = f".hf-write-probe/{stamp}-{run_id}.txt"
        body = (
            "Write-access probe. Safe to delete; the probe removes it itself.\n"
            f"utc={stamp} run={run_id}\n"
        ).encode()
        api.upload_file(
            path_or_fileobj=body,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=tok,
            commit_message=f"Write-access probe {stamp} (auto-deleted)",
        )
        if path_in_repo not in api.list_repo_files(
            args.repo_id, repo_type=args.repo_type, token=tok
        ):
            raise RuntimeError(f"upload reported success but {path_in_repo} is not listed")
        try:
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                token=tok,
                commit_message=f"Remove write-access probe {stamp}",
            )
            cleanup = "probe deleted"
        except Exception as exc:  # noqa: BLE001
            cleanup = f"PROBE FILE LEFT AT {path_in_repo} -- delete manually ({exc})"
        return f"wrote and verified {path_in_repo}; {cleanup}"

    def source() -> str:
        """Report both flavours of every tier, with the repo prefix each maps to.

        Both are uploaded: ``splits/<tier>`` is the quality-filtered
        train/validation/test data the model consumed, and ``by_project/<tier>`` is the
        tier before filtering and splitting, from which the splits can be re-derived.
        Paths are read from the staging tree, which already mirrors the repository.
        """
        root = args.source_root
        if not root.is_dir():
            raise RuntimeError(f"{root} is not a directory")

        def human(n: float) -> str:
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if abs(n) < 1024 or unit == "TB":
                    return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
                n /= 1024.0
            return f"{n:.1f}TB"

        rows, total = [], 0
        for tier in TIERS:
            for dirname, prefix in (
                (f"splits/{tier}", f"splits/{tier}"),
                (f"by_project/{tier}", f"by_project/{tier}"),
            ):
                d = root / dirname
                if not d.is_dir():
                    rows.append(f"{dirname:16s} ABSENT")
                    continue
                files = [f for f in d.rglob("*") if f.is_file()]
                size = sum(f.stat().st_size for f in files)
                total += size
                rows.append(
                    f"{dirname:16s} -> {prefix:18s} {len(files):>8,} files  {human(size):>9s}"
                )
        if not rows:
            raise RuntimeError(f"no tier directories found under {root}")
        return (
            f"{root}\n      "
            + "\n      ".join(rows)
            + f"\n      {'total':16s}    {human(total):>9s}"
        )

    probe.run("1. token present", token_present)
    probe.run("2. DNS resolves huggingface.co", dns)
    probe.run("3. HTTPS egress to huggingface.co", egress)
    probe.run("4. token authenticates", whoami)
    probe.run("5. repo readable", readable)
    if args.skip_write:
        print("[SKIP] 6. repo writable (--skip-write)")
    else:
        probe.run("6. repo writable", writable)
    if args.source_root is not None:
        probe.run("7. source tiers visible (both flavours)", source, critical=False)
    return probe.summary()


if __name__ == "__main__":
    sys.exit(main())
