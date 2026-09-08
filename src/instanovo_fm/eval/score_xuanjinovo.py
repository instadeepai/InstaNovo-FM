"""Score upstream XuanjiNovo de novo predictions against InstaNovo targets."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from instanovo.__init__ import console
from instanovo_fm.eval._predict_de_novo_common import (
    XUANJINOVO_BRACKETED_TO_UNIMOD,
    build_prediction_dataframe,
    save_predictions,
    score_predictions,
)
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

# Confidence floor
_MIN_SCORE = 1e-12

# Ordered patterns for recovering the join key from XuanjiNovo's title.
_ID_PATTERNS = (re.compile(r"pid=(\S+)"), re.compile(r"_SCANS_(\d+)"), re.compile(r"(\d+)\s*$"))


_CSV_METRIC_NAMES = {
    "peptide_recall": "pep_recall",
    "peptide_precision": "pep_prec",
    "aa_recall": "aa_recall",
    "aa_precision": "aa_prec",
    "aa_error_rate": "aa_er",
    "auc": "auc",
    "recall_at_0.05_fdr": "pep_recall_at_0.050_fdr",
}


def parse_prediction_id(title: str) -> Optional[str]:
    """Recover the ``prediction_id`` join key from a XuanjiNovo output title.

    Args:
        title: The ``title`` string from a ``denovo.tsv`` row.

    Returns:
        The parsed id **as a string** — an int index (``"5"``) in single-dataset mode or a composite
        ``"<dataset>:<index>"`` in combined mode — or ``None`` if no pattern matches. Kept as a string
        so both forms join uniformly against the sidecar.
    """
    for pattern in _ID_PATTERNS:
        match = pattern.search(str(title))
        if match:
            return str(match.group(1))
    return None


def load_xuanjinovo_predictions(tsv_path: str) -> pd.DataFrame:
    """Load a XuanjiNovo ``denovo.tsv`` into ``prediction_id, prediction, score`` columns.

    Args:
        tsv_path: Path to the tab-separated ``denovo.tsv`` (columns ``title, prediction, charge, score``).

    Returns:
        A DataFrame with one row per predicted spectrum: ``prediction_id`` (parsed from ``title``),
        ``prediction`` (peptide string), ``score`` (float confidence in ``[0, 1]``). Rows whose id
        cannot be parsed are dropped with a warning.

    Raises:
        KeyError: If the expected ``title`` / ``prediction`` / ``score`` columns are absent.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for required in ("title", "prediction", "score"):
        if required not in df.columns:
            raise KeyError(f"denovo.tsv is missing the '{required}' column (found {list(df.columns)}).")

    df["prediction_id"] = df["title"].map(parse_prediction_id)
    n_unparsed = int(df["prediction_id"].isna().sum())
    if n_unparsed:
        example = df.loc[df["prediction_id"].isna(), "title"].head(1).tolist()
        logger.warning(f"Dropping {n_unparsed:,} prediction rows whose title had no parseable id (e.g. {example}).")
        df = df[df["prediction_id"].notna()]
    df["prediction_id"] = df["prediction_id"].astype(str)
    df["prediction"] = df["prediction"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df[["prediction_id", "prediction", "score"]]


def build_canonical_dataframe(preds: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Join XuanjiNovo predictions onto the targets sidecar and build the canonical prediction frame.

    The sidecar is the universe of scored spectra; predictions are joined onto it. A spectrum with no
    prediction gets an empty ``predictions`` string and ``predictable=False`` (a miss under the true
    denominator). Confidence is stored as ``log_probs = log(score)`` so the shared scorer recovers it
    via ``exp(log_probs)``.

    Args:
        preds: XuanjiNovo predictions (``prediction_id, prediction, score``) from :func:`load_xuanjinovo_predictions`.
        sidecar: The targets sidecar (``prediction_id, targets, precursor_mz, precursor_charge, group``).

    Returns:
        A DataFrame in the canonical schema (see :data:`PREDICTION_CSV_COLUMNS`).
    """
    # Join on string ids so both single-dataset (int index) and combined (composite string) modes work.
    preds = preds.copy()
    preds["prediction_id"] = preds["prediction_id"].astype(str)
    sidecar = sidecar.copy()
    sidecar["prediction_id"] = sidecar["prediction_id"].astype(str)

    # TODO investigate
    # Keep the last prediction per id if the model emitted duplicates (defensive; expect 1:1).
    preds = preds.drop_duplicates(subset="prediction_id", keep="last")

    orphans = set(preds["prediction_id"]) - set(sidecar["prediction_id"])
    if orphans:
        logger.warning(f"{len(orphans):,} predicted ids are absent from the sidecar and will be ignored (first few: {sorted(orphans)[:5]}).")

    merged = sidecar.merge(preds, on="prediction_id", how="left")
    has_pred = merged["prediction"].notna()
    log_probs = np.log(np.clip(merged["score"].to_numpy(dtype=float), _MIN_SCORE, 1.0))

    records = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        predictable = bool(has_pred.iloc[i])
        records.append(
            {
                "prediction_id": str(row["prediction_id"]),
                "predictions": str(row["prediction"]) if predictable else "",
                "targets": str(row["targets"]) if row["targets"] is not None else "",
                "log_probs": float(log_probs[i]) if predictable else float("nan"),
                "predictable": predictable,
                "precursor_mz": float(row["precursor_mz"]),
                "precursor_charge": int(row["precursor_charge"]),
                "group": str(row["group"]),
            }
        )
    n_predicted = int(has_pred.sum())
    logger.info(
        f"Joined {n_predicted:,}/{len(merged):,} sidecar spectra to a XuanjiNovo prediction ({len(merged) - n_predicted:,} counted as misses)."
    )
    return build_prediction_dataframe(records)


def write_results_csv(
    results: Dict[str, Any],
    results_csv: str,
    *,
    run_name: str,
    model_label: str,
    num_beams: int,
    use_knapsack: bool,
    s3_handler: Optional[S3FileHandler] = None,
) -> None:
    """Append a single wide per-group results row to ``results_csv`` (InstaNovo predictor.py format).

    Mirrors ``AccelerateDeNovoPredictor.save_predictions``: metadata columns then, for each group,
    ``{group}_{metric}`` columns using InstaNovo's metric names (``pep_recall``, ``pep_prec``,
    ``aa_recall``, ``aa_prec``, ``aa_er``, ``auc``, ``pep_recall_at_0.050_fdr``). The row is appended
    to any existing CSV (read + concat), so a XuanjiNovo run slots in beside InstaNovo/other baselines and
    plots directly with ``scripts/plot_pep_recall_benchmark.py``. Overall metrics are not written here
    (matching predictor.py) — they live in the scorer's ``summary.json`` and the logs.

    Args:
        results: The :func:`score_predictions` result dict (must carry a ``"groups"`` block).
        results_csv: Destination CSV (local path or ``s3://`` URI).
        run_name: Run label (``run_name`` column).
        model_label: Model label (written to the ``instanovo_model`` column for row compatibility).
        num_beams: Beam width (``num_beams`` column).
        use_knapsack: Whether mass-controlled decoding was used (``use_knapsack`` column).
        s3_handler: Optional handler to reuse; a fresh one is created when not given.
    """
    row: Dict[str, Any] = {"run_name": run_name, "instanovo_model": model_label, "num_beams": num_beams, "use_knapsack": use_knapsack}
    groups = results.get("groups", {})
    for group_name, block in groups.items():
        for src, dst in _CSV_METRIC_NAMES.items():
            if src in block:
                row[f"{group_name}_{dst}"] = block[src]

    handler = s3_handler or S3FileHandler()
    local = handler.get_local_path(results_csv, missing_ok=True)
    if local is not None and os.path.exists(local):
        existing = pd.read_csv(local)
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, join="outer")
    else:
        out = pd.DataFrame([row])
    handler.upload_to_s3_wrapper(out.to_csv, results_csv, index=False)
    logger.info(f"Appended results row to {results_csv} ({len(groups)} group(s), {len(row) - 4} metric columns)")


def score(
    tsv_path: str,
    sidecar_path: str,
    output_csv: str,
    *,
    output_dir: Optional[str] = None,
    per_group: bool = True,
    results_csv: Optional[str] = None,
    run_name: str = "xuanjinovo",
    model_label: str = "xuanjinovo",
    num_beams: int = 40,
    use_knapsack: bool = True,
) -> Dict[str, Any]:
    """Convert a XuanjiNovo ``denovo.tsv`` to the canonical CSV and score it with the shared harness.

    Args:
        tsv_path: Path to the XuanjiNovo ``denovo.tsv``.
        sidecar_path: Path to the targets sidecar parquet from :mod:`convert_parquet_to_mgf`.
        output_csv: Path to write the canonical prediction CSV (local or ``s3://``).
        output_dir: Optional directory for the scorer's ``summary.json`` (overall + per-group).
        per_group: Compute per-group (per-species) metrics in addition to overall (default True). In a
            combined run ``group`` is the dataset/species name; in a single-dataset run it is whatever
            :func:`~instanovo_fm.eval._predict_de_novo_common._resolve_group` resolved.
        results_csv: If given, append the per-group wide results row here (InstaNovo predictor.py format).
        run_name: Run label for the results row.
        model_label: Model label for the results row (``instanovo_model`` column).
        num_beams: Beam width for the results row.
        use_knapsack: Mass-controlled-decoding flag for the results row.

    Returns:
        The :func:`score_predictions` result dict — ``{"overall": {...}}`` plus ``{"groups": {...}}``
        when ``per_group`` is set.
    """
    preds = load_xuanjinovo_predictions(tsv_path)
    sidecar = pd.read_parquet(sidecar_path)
    canonical = build_canonical_dataframe(preds, sidecar)
    save_predictions(canonical, output_csv)
    results: Dict[str, Any] = score_predictions(output_csv, output_dir, residue_remapping=XUANJINOVO_BRACKETED_TO_UNIMOD, per_group=per_group)
    if results_csv:
        write_results_csv(results, results_csv, run_name=run_name, model_label=model_label, num_beams=num_beams, use_knapsack=use_knapsack)
    return results


def main() -> None:
    """CLI entry point: score a XuanjiNovo denovo.tsv against the targets sidecar."""
    parser = argparse.ArgumentParser(description="Score upstream XuanjiNovo de novo predictions against InstaNovo targets.")
    parser.add_argument("--tsv_path", required=True, help="XuanjiNovo denovo.tsv output.")
    parser.add_argument("--sidecar_path", required=True, help="Targets sidecar parquet from convert_parquet_to_mgf.")
    parser.add_argument("--output_csv", required=True, help="Canonical prediction CSV to write (local or s3://).")
    parser.add_argument("--output_dir", default=None, help="Directory for summary.json (overall + per-group).")
    parser.add_argument("--results_csv", default=None, help="Append the per-group wide results row here (predictor.py format).")
    parser.add_argument("--run_name", default="xuanjinovo", help="Run label for the results row.")
    parser.add_argument("--model_label", default="xuanjinovo", help="Model label (instanovo_model column) for the results row.")
    parser.add_argument("--num_beams", type=int, default=40, help="Beam width, recorded in the results row.")
    parser.add_argument("--no_knapsack", action="store_true", help="Record use_knapsack=False (default True: PMC on).")
    args = parser.parse_args()
    score(
        args.tsv_path,
        args.sidecar_path,
        args.output_csv,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        run_name=args.run_name,
        model_label=args.model_label,
        num_beams=args.num_beams,
        use_knapsack=not args.no_knapsack,
    )


if __name__ == "__main__":
    main()
