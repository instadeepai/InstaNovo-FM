"""Convert InstaNovo validation parquet to MGF for the upstream XuanjiNovo model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from instanovo.__init__ import console
from instanovo_fm.eval._extract_embeddings_common import load_spectrum_dataframe
from instanovo_fm.eval._predict_de_novo_common import (
    XUANJINOVO_TO_UNIMOD,
    _resolve_group,
    build_model_vocab,
    filter_unpredictable_rows,
    reconcile_max_charge,
)
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

MAX_CHARGE = 10

XUANJINOVO_RESIDUES: List[str] = [
    "G",
    "A",
    "S",
    "P",
    "V",
    "T",
    "L",
    "I",
    "N",
    "D",
    "Q",
    "K",
    "E",
    "M",
    "H",
    "F",
    "R",
    "Y",
    "W",
    "C+57.021",  # cysteine, fixed carbamidomethyl
    "M+15.995",  # oxidised methionine (variable)
    "N+0.984",
    "Q+0.984",  # deamidation
    "+42.011",  # acetylation (N-term)
    "+43.006",  # carbamylation (N-term)
    "-17.027",  # loss of ammonia (N-term)
]


def build_title(prediction_id: Any) -> str:
    """Build the MGF ``TITLE`` that encodes the join key.

    Args:
        prediction_id: The join key — a 0-based int (single-dataset mode) or a composite
            ``<dataset>:<index>`` string (combined multi-dataset mode).

    Returns:
        A title of the form ``pid=<prediction_id>`` — parsed back by the scorer.
    """
    return f"pid={prediction_id}"


def _mgf_record(row: Any, prediction_id: Any, scan: int) -> Dict[str, Any]:
    """Build one pyteomics MGF spectrum dict from a spectra row (peaks sorted by ascending m/z).

    Args:
        row: A pandas row with ``mz_array``, ``intensity_array``, ``precursor_mz``, ``precursor_charge``.
        prediction_id: The join key encoded in the ``TITLE`` (int index, or composite string).
        scan: Globally-unique scan number for the MGF ``SCANS`` field.

    Returns:
        A ``{"m/z array", "intensity array", "params"}`` dict for :func:`pyteomics.mgf.write`.
    """
    mz = np.asarray(row["mz_array"], dtype=float)
    intensity = np.asarray(row["intensity_array"], dtype=float)
    order = np.argsort(mz)
    return {
        "m/z array": mz[order],
        "intensity array": intensity[order],
        "params": {
            "title": build_title(prediction_id),
            "pepmass": (float(row["precursor_mz"]),),
            "charge": f"{int(row['precursor_charge'])}+",
            "scans": str(scan),
        },
    }


def _filter_dataset(
    parquet_path: str,
    *,
    max_samples: Optional[int],
    max_charge: int,
    column_map: Optional[Dict[str, str]],
) -> tuple[pd.DataFrame, bool]:
    """Load a parquet dataset and drop rows XuanjiNovo structurally cannot handle.

    Args:
        parquet_path: Input parquet file(s) / glob, or an ``s3://`` URI.
        max_samples: Optional cap on the number of spectra loaded.
        max_charge: Maximum precursor charge (reconciled to the model's limit).
        column_map: Optional source→canonical column renaming.

    Returns:
        Tuple ``(filtered_df, has_targets)`` — the filtered, reindexed frame and whether it carries a
        ``sequence`` column.

    Raises:
        ValueError: If no rows survive the filter.
    """
    df = load_spectrum_dataframe(parquet_path, max_samples, column_mapping=column_map)
    has_targets = "sequence" in df.columns
    if not has_targets:
        logger.warning("No 'sequence' column found — sidecar targets will be empty (de novo mode).")
    model_vocab = build_model_vocab(XUANJINOVO_RESIDUES, XUANJINOVO_TO_UNIMOD)
    effective_max_charge = reconcile_max_charge(max_charge, MAX_CHARGE, "XuanjiNovo")
    df = filter_unpredictable_rows(df, max_charge=effective_max_charge, model_vocab=model_vocab, model_name="XuanjiNovo").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No spectra left to convert from {parquet_path} after filtering.")
    return df, has_targets


def _sidecar_rows(df: pd.DataFrame, has_targets: bool, *, dataset_name: Optional[str]) -> List[Dict[str, Any]]:
    """Build sidecar rows for a filtered frame.

    Args:
        df: The filtered spectra frame.
        has_targets: Whether ``df`` has a ``sequence`` column.
        dataset_name: If given (combined mode), ``prediction_id`` is the composite ``<name>:<i>`` and
            ``group`` is the dataset name; otherwise (single mode) ``prediction_id`` is the int index
            and ``group`` comes from :func:`_resolve_group`.

    Returns:
        One dict per row with the sidecar columns.
    """
    columns = list(df.columns)
    targets = [str(t) if t is not None else "" for t in df["sequence"]] if has_targets else [""] * len(df)
    rows: List[Dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        rows.append(
            {
                "prediction_id": f"{dataset_name}:{i}" if dataset_name is not None else i,
                "targets": targets[i],
                "precursor_mz": float(row["precursor_mz"]),
                "precursor_charge": int(row["precursor_charge"]),
                "group": dataset_name if dataset_name is not None else _resolve_group(row, columns),
            }
        )
    return rows


def convert(
    parquet_path: str,
    mgf_path: str,
    sidecar_path: str,
    *,
    max_samples: Optional[int] = None,
    max_charge: int = MAX_CHARGE,
    column_map: Optional[Dict[str, str]] = None,
) -> int:
    """Convert one InstaNovo parquet dataset to an MGF file + a targets sidecar parquet.

    Single-dataset mode: ``prediction_id`` is the 0-based row index and ``group`` is resolved from
    ``frag_type`` / ``experiment_name``. For a combined multi-species run (one job over all datasets)
    use :func:`convert_combined`, which assigns composite ids and per-dataset groups.

    Args:
        parquet_path: Path to the input parquet file(s) / glob, or an ``s3://`` URI.
        mgf_path: Local path to write the MGF file to.
        sidecar_path: Local path to write the targets sidecar parquet to.
        max_samples: Optional cap on the number of spectra loaded (useful for smoke tests).
        max_charge: Maximum precursor charge XuanjiNovo can encode (reconciled to the model's limit).
        column_map: Optional source→canonical column renaming passed to
            :func:`load_spectrum_dataframe` (e.g. an inference config's ``column_map``).

    Returns:
        The number of spectra written (rows surviving the up-front filter).
    """
    from pyteomics import mgf

    df, has_targets = _filter_dataset(parquet_path, max_samples=max_samples, max_charge=max_charge, column_map=column_map)
    Path(mgf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    mgf.write((_mgf_record(df.iloc[i], i, i) for i in range(len(df))), output=mgf_path, use_numpy=True)
    pd.DataFrame(_sidecar_rows(df, has_targets, dataset_name=None)).to_parquet(sidecar_path, index=False)
    logger.info(f"Wrote {len(df):,} spectra to {mgf_path}")
    logger.info(f"Wrote targets sidecar ({len(df):,} rows) to {sidecar_path}")
    return len(df)


# TODO rename sidecar
def convert_combined(
    datasets: List[tuple],
    mgf_path: str,
    sidecar_path: str,
    *,
    max_samples: Optional[int] = None,
    max_charge: int = MAX_CHARGE,
    column_map: Optional[Dict[str, str]] = None,
) -> int:
    """Convert many datasets into ONE MGF + ONE combined sidecar for a single multi-species run.

    Each spectrum gets a composite ``prediction_id`` (``<dataset_name>:<local_index>``) encoded in its
    MGF ``TITLE``, so ids stay globally unique across datasets and the single ``denovo.tsv`` the model
    emits joins unambiguously back to the combined sidecar. ``group`` is the dataset name, which the
    scorer uses for per-group (per-species) metrics. ``SCANS`` is a global running index so the MGF
    has no duplicate scan numbers. Datasets are streamed one at a time (only one frame in memory).

    Args:
        datasets: Ordered list of ``(dataset_name, parquet_path)`` pairs. ``dataset_name`` should be
            the canonical ``result_name`` (e.g. ``hela_qc``) so it matches the benchmark/plot columns.
        mgf_path: Local path for the combined MGF.
        sidecar_path: Local path for the combined targets sidecar parquet.
        max_samples: Optional per-dataset cap on spectra loaded.
        max_charge: Maximum precursor charge (reconciled to the model's limit).
        column_map: Optional source→canonical column renaming.

    Returns:
        Total number of spectra written across all datasets.

    Raises:
        ValueError: If no rows survive filtering across all datasets.
    """
    from pyteomics import mgf

    Path(mgf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)

    sidecar_rows: List[Dict[str, Any]] = []
    scan = 0
    with open(mgf_path, "w") as handle:
        for name, parquet_path in datasets:
            logger.info(f"[{name}] converting {parquet_path}")
            df, has_targets = _filter_dataset(parquet_path, max_samples=max_samples, max_charge=max_charge, column_map=column_map)
            offset = scan
            mgf.write((_mgf_record(df.iloc[i], f"{name}:{i}", offset + i) for i in range(len(df))), output=handle, use_numpy=True)
            sidecar_rows.extend(_sidecar_rows(df, has_targets, dataset_name=name))
            scan += len(df)
            logger.info(f"[{name}] wrote {len(df):,} spectra (running total {scan:,})")

    if not sidecar_rows:
        raise ValueError("No spectra left to convert across all datasets after filtering.")
    pd.DataFrame(sidecar_rows).to_parquet(sidecar_path, index=False)
    logger.info(f"Wrote {scan:,} spectra to {mgf_path}")
    logger.info(f"Wrote combined sidecar ({len(sidecar_rows):,} rows, {len(datasets)} datasets) to {sidecar_path}")
    return scan


def main() -> None:
    """CLI entry point: convert one dataset, or combine many, to MGF + targets sidecar.

    Single dataset: ``--parquet_path``. Combined multi-species (one job): ``--datasets name=path ...``
    (composite ids + per-dataset groups). Exactly one of the two must be given.
    """
    parser = argparse.ArgumentParser(description="Convert InstaNovo parquet to MGF for upstream XuanjiNovo inference.")
    parser.add_argument("--parquet_path", default=None, help="Single-dataset input parquet file(s) / glob, or s3:// URI.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Combined mode: one or more '<result_name>=<parquet path/glob>' pairs merged into one MGF + sidecar.",
    )
    parser.add_argument("--mgf_path", required=True, help="Output MGF path.")
    parser.add_argument("--sidecar_path", required=True, help="Output targets sidecar parquet path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on spectra loaded per dataset (smoke tests).")
    parser.add_argument("--max_charge", type=int, default=MAX_CHARGE, help="Maximum precursor charge.")
    args = parser.parse_args()

    if bool(args.datasets) == bool(args.parquet_path):
        parser.error("Provide exactly one of --parquet_path (single) or --datasets (combined).")

    if args.datasets:
        pairs = [tuple(spec.split("=", 1)) for spec in args.datasets]
        if any(len(p) != 2 or not p[0] or not p[1] for p in pairs):
            parser.error("Each --datasets entry must be '<name>=<path>'.")
        convert_combined(pairs, args.mgf_path, args.sidecar_path, max_samples=args.max_samples, max_charge=args.max_charge)
    else:
        convert(args.parquet_path, args.mgf_path, args.sidecar_path, max_samples=args.max_samples, max_charge=args.max_charge)


if __name__ == "__main__":
    main()
