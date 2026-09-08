"""Container entrypoint for the upstream XuanjiNovo de novo benchmark.

Usage:
    python -m instanovo_fm.eval.run_xuanjinovo <input_mgf> <output_prefix>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import urlparse

from instanovo.__init__ import console
from instanovo_fm.utils.checkpoints import S3_NOT_CONFIGURED_HINT, resolve_checkpoint
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CHECKPOINT_URL = "https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_100M_massnet.ckpt"
WORK_DIR = Path("/tmp/xuanjinovo")
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
LOG_SUFFIXES = ("*.log", "*_cl_out.txt")
TSV_NAME = "denovo.tsv"


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, accepting ``1`` / ``true`` / ``yes`` (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Resolve the run configuration from positional arguments, falling back to the environment.

    Args:
        argv: Argument vector to parse (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: If the input MGF or output prefix is supplied by neither route.
    """
    parser = argparse.ArgumentParser(prog="run_xuanjinovo", description="Run the upstream XuanjiNovo de novo benchmark.")
    parser.add_argument("input_mgf", nargs="?", default=os.environ.get("XUANJINOVO_INPUT"), help="Input MGF: local path or s3:// URI.")
    parser.add_argument("output_prefix", nargs="?", default=os.environ.get("XUANJINOVO_OUTPUT"), help="Output prefix: local path or s3:// URI.")
    parser.add_argument("--n_beams", type=int, default=int(os.environ.get("XUANJINOVO_N_BEAMS", "40")))
    parser.add_argument("--mass_control_tol", type=float, default=float(os.environ.get("XUANJINOVO_MASS_CONTROL_TOL", "0.1")))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("XUANJINOVO_BATCH_SIZE", "64")))
    parser.add_argument("--gpu", default=os.environ.get("XUANJINOVO_GPU", "0"))
    parser.add_argument("--checkpoint_path", default=os.environ.get("XUANJINOVO_CKPT_S3"), help="Checkpoint s3:// URI or local path.")
    parser.add_argument("--checkpoint_url", default=os.environ.get("XUANJINOVO_CKPT_URL", CHECKPOINT_URL))
    parser.add_argument("--app_dir", default=os.environ.get("XUANJINOVO_APP_DIR", "/app"), help="Working directory holding the upstream source.")
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path(os.environ.get("XUANJINOVO_WORK_DIR", str(WORK_DIR))),
        help="Directory for upstream's output. Wiped at the start of every run.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(os.environ.get("XUANJINOVO_DATA_DIR", str(DATA_DIR))),
        help="Directory downloaded input MGFs are written to.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path(os.environ.get("XUANJINOVO_CKPT_DIR", str(CHECKPOINT_DIR))),
        help="Directory downloaded checkpoints are written to.",
    )
    parser.add_argument(
        "--allow_local_fallback",
        action="store_true",
        default=_env_flag("XUANJINOVO_ALLOW_LOCAL_FALLBACK"),
        help="Keep the local copy instead of failing when the remote write does not land. Off by default: this container is "
        "ephemeral, so a local-only result is a lost result.",
    )
    args = parser.parse_args(argv)

    if not args.input_mgf:
        parser.error("input MGF is required as argument 1 or XUANJINOVO_INPUT (local path or s3:// URI)")
    if not args.output_prefix:
        parser.error("output prefix is required as argument 2 or XUANJINOVO_OUTPUT (local path or s3:// URI)")
    return args


def check_destination(output_prefix: str, *, allow_local_fallback: bool) -> None:
    """Fail before inference if the output prefix cannot be written to.

    Args:
        output_prefix: Destination prefix, local or ``s3://``.
        allow_local_fallback: If True, downgrade an unconfigured-S3 destination to a warning.

    Raises:
        RuntimeError: If the prefix is an ``s3://`` URI but S3 is not configured and no fallback is allowed.
    """
    if not output_prefix.startswith("s3://") or S3FileHandler.s3_enabled():
        return
    message = f"Output prefix '{output_prefix}' is an S3 URI but S3 is not configured. {S3_NOT_CONFIGURED_HINT}"
    if not allow_local_fallback:
        raise RuntimeError(message)
    logger.warning(f"{message} Results will be kept locally only.")


def resolve_input_mgf(input_mgf: str, data_dir: Path, s3: S3FileHandler) -> Path:
    """Resolve the input MGF to a local file, downloading it when it lives on S3.

    Args:
        input_mgf: Local path or ``s3://bucket/key`` URI.
        data_dir: Directory that inputs are downloaded into.
        s3: The handler used for the download.

    Returns:
        Local path to the MGF.

    Raises:
        RuntimeError: If the URI is on S3 but S3 is not configured, or the download produced no data.
        FileNotFoundError: If a local path does not exist.
    """
    if not input_mgf.startswith("s3://"):
        local = Path(input_mgf)
        if not local.exists():
            raise FileNotFoundError(f"Input MGF not found: {local}")
        logger.info(f"Using local input MGF at {local}")
        return local

    if not S3FileHandler.s3_enabled():
        raise RuntimeError(f"Input MGF '{input_mgf}' is an S3 URI but S3 is not configured. {S3_NOT_CONFIGURED_HINT}")

    dest = data_dir / Path(input_mgf).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading input MGF from {input_mgf} → {dest}")
    s3.download(input_mgf, str(dest))
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Download of {input_mgf} produced no data at {dest}.")
    logger.info("Download complete")
    return dest


def run_inference(args: argparse.Namespace, mgf: Path, checkpoint: Path, out_dir: Path) -> None:
    """Run upstream XuanjiNovo de novo inference as a subprocess.

    Args:
        args: Parsed run configuration.
        mgf: Local path to the input MGF.
        checkpoint: Local path to the model checkpoint.
        out_dir: Directory passed as ``--output``; upstream creates a timestamped subdirectory inside it.

    Raises:
        subprocess.CalledProcessError: If upstream exits non-zero.
    """
    command = [
        sys.executable,
        "-m",
        "XuanjiNovo.XuanjiNovo",
        "--mode=denovo",
        f"--model={checkpoint}",
        f"--peak_path={mgf}",
        f"--n_beams={args.n_beams}",
        "--pmc-enable",
        f"--mass_control_tol={args.mass_control_tol}",
        f"--batch_size={args.batch_size}",
        f"--gpu={args.gpu}",
        f"--output={out_dir}",
    ]
    logger.info(f"Running XuanjiNovo de novo: n_beams={args.n_beams} pmc_tol={args.mass_control_tol} batch_size={args.batch_size} gpu={args.gpu}")
    subprocess.run(command, cwd=args.app_dir, check=True)


def find_denovo_tsv(out_dir: Path) -> Path:
    """Locate the ``denovo.tsv`` upstream wrote under ``out_dir``.

    Args:
        out_dir: The directory passed to upstream as ``--output``.

    Returns:
        Path to the predictions TSV.

    Raises:
        FileNotFoundError: If upstream produced no TSV.
        RuntimeError: If more than one TSV is present, which the per-run wipe should make impossible.
    """
    matches = sorted(out_dir.rglob(TSV_NAME))
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one {TSV_NAME} under {out_dir}, found {len(matches)}: {[str(m) for m in matches]}. "
            "The output directory is wiped per run, so this means it was shared or pre-populated."
        )

    if not matches:
        contents = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*"))
        raise FileNotFoundError(f"{TSV_NAME} not found under {out_dir}. Contents: {contents}")

    return matches[0]


def _destination_exists(destination: str, s3: S3FileHandler) -> bool:
    """Check that a written file actually landed, on S3 or on disk."""
    if destination.startswith("s3://"):
        return s3.s3 is not None and bool(s3.s3.exists(destination))
    return Path(destination).exists()


def write_result(source: Path, destination: str, s3: S3FileHandler) -> None:
    """Copy ``source`` to ``destination`` (local or ``s3://``) and confirm it landed.

    Uses :meth:`S3FileHandler.upload_to_s3_wrapper`, which writes local destinations directly and
    uploads ``s3://`` ones. That method's underlying ``upload`` swallows exceptions by design (a
    transient object-store error must not kill a training run), so success is verified explicitly
    rather than inferred from the absence of a raise.

    Args:
        source: Local file to copy.
        destination: Target path, local or ``s3://``.
        s3: The handler used for the write.

    Raises:
        RuntimeError: If the destination does not exist after the write.
    """
    logger.info(f"Writing {source} → {destination}")
    s3.upload_to_s3_wrapper(lambda target: shutil.copy(str(source), target), destination)
    if not _destination_exists(destination, s3):
        raise RuntimeError(f"Write of {source} to {destination} did not land. The object store reported no object at that key.")


def publish(tsv: Path, out_dir: Path, output_prefix: str, s3: S3FileHandler, *, allow_local_fallback: bool) -> str:
    """Write the predictions TSV and the run logs to the output prefix.

    The TSV is the deliverable and is verified. Logs are best-effort: a debugging artefact failing to
    upload should not discard a completed benchmark run.

    Args:
        tsv: Local predictions TSV.
        out_dir: Directory to scan for run logs.
        output_prefix: Destination prefix, local or ``s3://``.
        s3: The handler used for the writes.
        allow_local_fallback: If True, a failed TSV write warns and keeps the local copy.

    Returns:
        The path the TSV was published to, or the local path when a fallback was taken.

    Raises:
        RuntimeError: If the TSV write fails and no fallback is allowed.
    """
    prefix = output_prefix.rstrip("/")
    tsv_destination = f"{prefix}/{TSV_NAME}"
    try:
        write_result(tsv, tsv_destination, s3)
        published = tsv_destination
    except Exception as exc:
        if not allow_local_fallback:
            raise
        logger.warning(f"Publishing {TSV_NAME} to {tsv_destination} failed ({exc}); keeping the local copy at {tsv}.")
        published = str(tsv)

    for log_file in _collect_logs(out_dir):
        try:
            write_result(log_file, f"{prefix}/{log_file.name}", s3)
        except Exception as exc:  # noqa: BLE001 - logs are best-effort; one failure must not discard the run
            logger.warning(f"Could not publish log {log_file.name}: {exc}")

    return published


def _collect_logs(out_dir: Path) -> List[Path]:
    """Return the run-log files upstream wrote under ``out_dir``."""
    logs: List[Path] = []
    for pattern in LOG_SUFFIXES:
        logs.extend(p for p in out_dir.rglob(pattern) if p.is_file())
    return sorted(logs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark end to end.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    s3 = S3FileHandler()

    check_destination(args.output_prefix, allow_local_fallback=args.allow_local_fallback)

    # Upstream appends a fresh %Y%m%d%H%M%S subdirectory to --output on every invocation and never
    # clears the parent, so re-running in one container would accumulate a subdirectory per run.
    # Starting from an empty working directory keeps this run's output the only one present.
    # work_dir is by default a tmp dir, final outputs are saved to XUANJINOVO_OUTPUT
    out_dir: Path = args.work_dir / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    mgf = resolve_input_mgf(args.input_mgf, args.data_dir, s3)
    default_checkpoint = args.checkpoint_dir / (Path(urlparse(args.checkpoint_url).path).name or "model.ckpt")
    checkpoint = resolve_checkpoint(
        args.checkpoint_path or str(default_checkpoint),
        args.checkpoint_url,
        not_found_hint=f"  {CHECKPOINT_URL}",
        cache_dir=args.checkpoint_dir,
    )

    run_inference(args, mgf, checkpoint, out_dir)

    tsv = find_denovo_tsv(out_dir)
    published = publish(tsv, out_dir, args.output_prefix, s3, allow_local_fallback=args.allow_local_fallback)
    logger.info(f"Done. Predictions at {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
