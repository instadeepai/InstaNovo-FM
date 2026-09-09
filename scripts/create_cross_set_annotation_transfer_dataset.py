#!/usr/bin/env python3
r"""Build a Kostas-protocol cross-set parquet from paired ACFM/LCFM folders.

File-per-file ACFM−LCFM diff + top-N peptide anchors::

    uv run python scripts/create_cross_set_annotation_transfer_dataset.py \
        --query-dir /data/acfm/PXD074343 \
        --library-dir /data/lcfm/PXD074343 \
        --top-n-peptides 10 \
        --output /data/cross_set/PXD074343_kostas.parquet

Omit ``--file`` to process every basename-matched pair in the two folders.
Pass ``--top-n-peptides all`` to use every distinct peptide per file pair as an
anchor instead of a fixed top-N shortlist.
"""

from __future__ import annotations

import argparse
import json

from instanovo_fm.eval.cross_set_dataset import (
    build_kostas_protocol_parquet,
    discover_paired_files,
    infer_project_id_from_path,
)


def _parse_top_n_peptides(value: str) -> int | None:
    """Parse --top-n-peptides, allowing the literal "all" to mean no cap."""
    if value.strip().lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--top-n-peptides must be a positive integer or 'all'")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and write the combined Kostas-protocol parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True, help="Folder of unlabeled query spectra (ACFM)")
    parser.add_argument("--library-dir", required=True, help="Folder of labeled LCFM spectra (full LCFM registry)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output combined parquet path (required unless --list-files is set)",
    )
    parser.add_argument(
        "--overlap-key",
        default="scan",
        choices=["scan", "usi", "source_file"],
        help="Key used for the per-file ACFM−LCFM diff (default: scan with source_file prefix)",
    )
    parser.add_argument("--project-id", default=None, help="PXD id for synthetic USI when missing")
    parser.add_argument(
        "--file",
        default=None,
        help="Basename of one paired file (e.g. 477-1.mzML.ipc). Omit to run all basename-matched pairs.",
    )
    parser.add_argument(
        "--top-n-peptides",
        type=_parse_top_n_peptides,
        default=10,
        help="Number of LCFM-valid peptides to use as retrieval anchors, or 'all' for no cap "
        "(default: 10)",
    )
    parser.add_argument(
        "--valid-glob",
        default=None,
        help="Optional LCFM-valid parquet glob for peptide ranking "
        "(e.g. <data-root>/lcfm_splits/valid*.parquet). "
        "If unset or no match for the file, rank on the LCFM file itself.",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print the basename-matched ACFM/LCFM pair filenames (one per line) and exit, "
        "without building any parquet. Used to drive a per-file loop that scopes retrieval "
        "to one file pair at a time (avoids mixing library anchors across files).",
    )
    args = parser.parse_args(argv)

    if args.list_files:
        pairs = discover_paired_files(args.query_dir, args.library_dir)
        for acfm_path, _ in pairs:
            print(acfm_path.name)  # noqa: T201
        return 0

    if args.output is None:
        parser.error("--output is required unless --list-files is set")

    project_id = args.project_id or infer_project_id_from_path(args.query_dir) or infer_project_id_from_path(args.library_dir)

    summary = build_kostas_protocol_parquet(
        acfm_dir=args.query_dir,
        lcfm_dir=args.library_dir,
        output_path=args.output,
        overlap_key=args.overlap_key,
        project_id=project_id,
        top_n_peptides=args.top_n_peptides,
        valid_glob=args.valid_glob,
        file_name=args.file,
    )
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
