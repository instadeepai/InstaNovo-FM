"""Shared helpers for baseline de novo peptide prediction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

# parents[1] is the package root. It was parents[2] internally, where this module
# sat one level deeper at instanovo_fm/eval/; the "foundational"
# level is gone here, so the walk is one shorter.
_DEFAULT_RESIDUE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "residues" / "default.yaml"

PREDICTION_CSV_COLUMNS: List[str] = [  # TODO can we pull these csv column names from a model predict.py/general predictor.py?
    "prediction_id",
    "predictions",
    "targets",
    "log_probs",
    "predictable",
    "precursor_mz",
    "precursor_charge",
    "group",
]

# Casanovo detokenises to named-modification ProForma (e.g. "C[Carbamidomethyl]", "[Acetyl]-").
CASANOVO_TO_UNIMOD: dict[str, str] = {
    "C[Carbamidomethyl]": "C[UNIMOD:4]",
    "M[Oxidation]": "M[UNIMOD:35]",
    "N[Deamidated]": "N[UNIMOD:7]",
    "Q[Deamidated]": "Q[UNIMOD:7]",
    "[Acetyl]": "[UNIMOD:1]",
    "[Carbamyl]": "[UNIMOD:5]",
    "[Ammonia-loss]": "[UNIMOD:385]",
    "[+25.980265]": "[UNIMOD:5][UNIMOD:385]",
}

# XuanjiNovo writes modifications as bracketed mass offsets (e.g. "C[+57.021]",
# "M[+15.995]", "[+42.011]"), both in the upstream model's denovo.tsv that
# score_xuanjinovo reads and in the residue vocabulary convert_parquet_to_mgf
# filters against, so one table serves both. Keep XUANJINOVO_RESIDUES in the
# same notation: build_model_vocab canonicalises through this mapping, and a
# residue it cannot look up silently drops out of the vocabulary.
XUANJINOVO_BRACKETED_TO_UNIMOD: dict[str, str] = {
    "C[+57.021]": "C[UNIMOD:4]",
    "M[+15.995]": "M[UNIMOD:35]",
    "N[+0.984]": "N[UNIMOD:7]",
    "Q[+0.984]": "Q[UNIMOD:7]",
    "[+42.011]": "[UNIMOD:1]",  # acetylation (N-term)
    "[+43.006]": "[UNIMOD:5]",  # carbamylation (N-term)
    "[-17.027]": "[UNIMOD:385]",  # loss of ammonia (N-term)
    "[+43.006-17.027]": "[UNIMOD:5][UNIMOD:385]",
}

MODEL_SCORING: dict[str, dict[str, Any]] = {
    "casanovo": {"residue_remapping": CASANOVO_TO_UNIMOD},
    "xuanjinovo_upstream": {"residue_remapping": XUANJINOVO_BRACKETED_TO_UNIMOD},
}

# Unmatchable mass assigned to target residues absent from the scoring residue set, so such targets
# are still scored (as guaranteed misses) rather than dropped. Far outside any real residue mass.
_SENTINEL_MASS = 1_000_000.0  # TODO can we not just exclude?


def _resolve_group(row: Any, columns: List[str]) -> str:
    """Pick a group label for a spectrum: frag_type, else experiment_name, else 'all'.

    Args:
        row: A pandas row.
        columns: The DataFrame's column names.

    Returns:
        The resolved group label.
    """
    for col in ("frag_type", "experiment_name"):
        if col in columns:
            value = row[col]
            if value is not None and str(value).strip() not in ("", "None", "nan"):
                return str(value)
    return "all"


def log_supported_residues(residues: dict[str, float], model_name: str) -> None:
    """Log the residue vocabulary a model supports (standard amino acids + modifications).

    Args:
        residues: The model's residues
        model_name: The model name
    """
    standard = sorted(r for r in residues if len(r) == 1 and r.isalpha())
    mods = [r for r in residues if not (len(r) == 1 and r.isalpha())]
    logger.info(f"{model_name} supports {len(residues)} residues: {len(standard)} standard amino acids + {len(mods)} modifications")
    logger.info(f"  standard amino acids: {standard}")
    logger.info(f"  modifications:        {mods}")


def build_prediction_dataframe(records: List[dict[str, Any]]) -> pd.DataFrame:
    """Assemble per-spectrum prediction records into a DataFrame with the canonical column order.

    Args:
        records: One dict per spectrum, keyed by (a subset of) ``PREDICTION_CSV_COLUMNS``.

    Returns:
        A DataFrame with the canonical columns first, then any extra keys, sorted by prediction_id.
    """
    df = pd.DataFrame(records)
    if "prediction_id" in df.columns:
        df = df.sort_values("prediction_id").reset_index(drop=True)
    ordered = [c for c in PREDICTION_CSV_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


def save_predictions(df: pd.DataFrame, output_path: str, *, s3_handler: Optional[S3FileHandler] = None) -> None:
    """Write a prediction DataFrame to ``output_path`` as CSV (local path or ``s3://`` URI).

    Delegates to the git-tracked :meth:`S3FileHandler.upload_to_s3_wrapper`, which creates parent
    dirs for local paths and writes-then-uploads for ``s3://`` destinations — so the same call works
    locally and on AIchor (where per-dataset outputs go to S3).

    Args:
        df: The prediction DataFrame.
        output_path: Local path or ``s3://`` URI to write to.
        s3_handler: Optional handler to reuse (callers writing many files, e.g. the multi-dataset
            pipeline, pass one to avoid constructing an s3fs client per call). A fresh one is made
            when not provided.
    """
    handler = s3_handler or S3FileHandler()
    handler.upload_to_s3_wrapper(df.to_csv, output_path, index=False)
    logger.info(f"Wrote {len(df):,} predictions to {output_path}")


def _build_residue_set(
    residue_config: Optional[str],
    *,
    residue_remapping: Optional[dict[str, str]] = None,
    extra_masses: Optional[dict[str, float]] = None,
) -> ResidueSet:
    """Build a :class:`ResidueSet` from a residue-masses YAML, with an optional residue remapping.

    Args:
        residue_config: Path to the residue-masses YAML (defaults to ``configs/residues/default.yaml``).
        residue_remapping: Optional model-notation → canonical-UNIMOD token remapping. Used both by
            :meth:`ResidueSet.get_mass` (per-token) and by :func:`_canonicalize_peptide` to rewrite
            a baseline's predictions to the canonical notation before scoring.
        extra_masses: Optional extra residue→mass entries merged into the masses table (e.g. sentinel
            masses so target residues absent from the YAML are still scorable — as guaranteed misses).

    Returns:
        The configured :class:`ResidueSet`.
    """
    import yaml

    with open(residue_config or _DEFAULT_RESIDUE_CONFIG) as f:
        residue_masses = yaml.safe_load(f)["residues"]
    if extra_masses:
        residue_masses = {**residue_masses, **extra_masses}
    return ResidueSet(residue_masses=residue_masses, residue_remapping=residue_remapping)


# TODO refactor to support aa-level and precision metrics
def _canonicalize_peptide(peptide: str, residue_set: ResidueSet) -> str:
    """Rewrite a peptide from a model's own notation to canonical UNIMOD using the residue set.

    Tokenises with the residue set (robust to modification syntax) and maps each token through the
    set's ``residue_remapping``. Unlike a raw string replace this is order-independent and splits
    compound offsets correctly (e.g. XuanjiNovo's ``+43.006-17.027`` → two tokens), and the result
    re-tokenises cleanly so *string*-based metrics (AA error rate) match — the mass-based metrics are
    already remapping-aware via ``get_mass``.

    A token whose mapped form expands to a residue outside the scoring set's vocab means the peptide
    contains something the set cannot represent, so the whole peptide is treated as unpredicted and an
    empty string is returned.

    This will fail for amino acid and precision type metrics.

    Args:
        peptide: The predicted peptide in the model's notation (empty string if unpredicted).
        residue_set: The scoring residue set carrying the ``residue_remapping``.

    Returns:
        The peptide rewritten to canonical UNIMOD notation, or ``""`` if it contains an unknown residue.
    """
    canonical_tokens: List[str] = []
    for token in residue_set.tokenize(peptide):
        mapped = residue_set.residue_remapping.get(token, token)
        # A remapping may expand to several residues (e.g. "[+25.980265]" → "[UNIMOD:5][UNIMOD:385]"),
        # which is never a single vocab entry — re-tokenise and validate each residue it produces.
        if any(sub not in residue_set.vocab for sub in residue_set.tokenize(mapped)):
            return ""
        canonical_tokens.append(mapped)
    return "".join(canonical_tokens)


def _collapse_il(token: str) -> str:
    """Collapse the isoleucine token to leucine (mass-equivalent)."""
    return "L" if token == "I" else token


def build_model_vocab(model_residues: Iterable[str], residue_remapping: dict[str, str], *, residue_config: Optional[str] = None) -> set[str]:
    """Build the set of canonical (UNIMOD, I/L-collapsed) residues a model can emit.

    Each model residue (in the model's own notation) is canonicalised to UNIMOD via
    ``residue_remapping`` and re-tokenised, so compound N-terminal mods contribute their individual
    parts. ``I`` is collapsed to ``L`` (mass-equivalent) so I-containing targets are not spuriously
    flagged as out of vocabulary.

    Args:
        model_residues: The model's residue vocabulary in its own notation.
        residue_remapping: The model-notation → UNIMOD remapping (see :data:`MODEL_SCORING`).
        residue_config: Path to the residue-masses YAML (defaults to ``default.yaml``).

    Returns:
        The set of canonical residues the model can produce.
    """
    residue_set = _build_residue_set(residue_config, residue_remapping=residue_remapping)
    vocab: set[str] = set()
    for residue in model_residues:  # TODO make more efficient?
        for token in residue_set.tokenize(_canonicalize_peptide(str(residue), residue_set)):
            vocab.add(_collapse_il(token))
    return vocab


def targets_in_model_vocab(
    targets: Iterable[str],
    model_vocab: set[str],
    *,
    residue_config: Optional[str] = None,
) -> tuple[List[bool], Counter[str]]:
    """Flag, per target, whether every residue is one the model can emit (I/L-collapsed).

    A target with residues outside the model's vocabulary (e.g. a phosphopeptide for a model without
    phospho residues, or an unmodified cysteine for a model that only has carbamidomethyl-C) is one
    the model structurally cannot get right — the predictor drops these (no decode) and they count as
    misses under the *true* denominator. Empty targets (de novo / unlabelled) are treated as in-vocab
    (nothing to fail on).

    Args:
        targets: Target peptide strings.
        model_vocab: The model's canonical residue set (from :func:`build_model_vocab`).
        residue_config: Path to the residue-masses YAML (defaults to ``default.yaml``).

    Returns:
        Tuple ``(mask, oov_counts)`` — ``mask[i]`` is True when target ``i`` is fully representable,
        and ``oov_counts`` tallies each offending residue across targets.
    """
    residue_set = _build_residue_set(residue_config)
    oov_counts: Counter[str] = Counter()
    mask: List[bool] = []
    for target in targets:
        if not target:
            mask.append(True)
            continue
        missing = {token for token in residue_set.tokenize(str(target)) if _collapse_il(token) not in model_vocab}
        oov_counts.update(missing)
        mask.append(not missing)
    return mask, oov_counts


def reconcile_max_charge(config_max_charge: Optional[int], model_max_charge: int, model_name: str) -> int:
    """Reconcile a configured ``max_charge`` against the model's own, mirroring the InstaNovo predictor.

    The precursor charge indexes a fixed-size charge embedding, so a spectrum with charge above the
    model's ``max_charge`` cannot be encoded. A config may further *restrict* the accepted charge, but
    never *raise* it above the model's architectural limit.

    Args:
        config_max_charge: The ``max_charge`` from the inference config (``None`` = no config override).
        model_max_charge: The model's own ``max_charge`` (from its checkpoint).
        model_name: Model name, for logging.

    Returns:
        The effective ``max_charge`` — the smaller of the two, or the model's value when unset.
    """
    if config_max_charge is None:
        return model_max_charge
    config_max_charge = int(config_max_charge)
    if config_max_charge > model_max_charge:
        logger.warning(
            f"{model_name}: configured max_charge={config_max_charge} exceeds the model's max_charge={model_max_charge}; "
            f"overwriting to the model's value ({model_max_charge})."
        )
        return model_max_charge
    return config_max_charge


def filter_unpredictable_rows(
    df: pd.DataFrame,
    *,
    max_charge: int,
    model_vocab: set[str],
    model_name: str,
    charge_column: str = "precursor_charge",
    target_column: str = "sequence",
) -> pd.DataFrame:
    """Drop rows the model structurally cannot handle, mirroring the InstaNovo predictor's upfront filter.

    Applied before any decoding; dropped rows are excluded from the output entirely, so they count
    toward neither scoring denominator. Two filters:

    * **charge** — precursor charge outside ``[1, max_charge]`` (or non-numeric): outside the model's
      charge-embedding range, so the spectrum cannot be encoded.
    * **residues** — a target using residues outside the model's emitted vocabulary (only when a
      ``target_column`` is present): the peptide is structurally unrepresentable by the model.

    Args:
        df: The loaded spectra DataFrame.
        max_charge: The effective maximum precursor charge (see :func:`reconcile_max_charge`).
        model_vocab: The model's canonical residue vocabulary (from :func:`build_model_vocab`).
        model_name: Model name, for logging.
        charge_column: Name of the precursor-charge column.
        target_column: Name of the target-sequence column (absent in de novo / unlabelled data).

    Returns:
        The filtered DataFrame with its index reset (possibly empty).
    """
    original_size = len(df)
    charge = pd.to_numeric(df[charge_column], errors="coerce")
    charge_ok = (charge >= 1) & (charge <= max_charge)
    n_charge_drop = int((~charge_ok).sum())

    if target_column in df.columns:
        targets = [str(t) if t is not None else "" for t in df[target_column]]
        target_ok_list, oov_counts = targets_in_model_vocab(targets, model_vocab)
        target_ok = pd.Series(target_ok_list, index=df.index)
    else:
        target_ok = pd.Series(True, index=df.index)
        oov_counts = Counter()
    n_target_drop = int((~target_ok).sum())

    filtered = df[charge_ok & target_ok].reset_index(drop=True)
    if len(filtered) < original_size:
        logger.warning(
            f"{model_name}: dropped {original_size - len(filtered):,}/{original_size:,} rows before decoding "
            f"({n_charge_drop:,} with charge outside [1, {max_charge}] + {n_target_drop:,} with target residues out of vocabulary)."
        )
        if oov_counts:
            detail = ", ".join(f"{residue} ({count})" for residue, count in oov_counts.most_common())
            logger.warning(f"  target residues outside {model_name}'s vocabulary (dropped): {detail}")
    return filtered


def _targets_in_vocab(targets: pd.Series, residue_set: ResidueSet) -> tuple[pd.Series, Counter[str]]:
    """Flag targets whose residues are all in the scoring residue set, and tally the offenders.

    Targets carrying modifications outside the residue vocabulary (e.g. exotic UNIMOD mods present
    in some datasets) cannot be scored against the fixed residue masses — the mass-based matching
    would raise ``KeyError`` — and a baseline model cannot represent them anyway. This flags them so
    the caller can exclude them from scoring (with a logged breakdown) while keeping the full CSV.

    Args:
        targets: Series of target peptide strings.
        residue_set: The scoring residue set (from :func:`_build_residue_set`).

    Returns:
        Tuple ``(mask, oov_counts)`` where ``mask`` is a boolean Series (True where every target
        residue is in the residue set) and ``oov_counts`` maps each out-of-vocabulary residue to the
        number of excluded targets that contain it.
    """
    oov_counts: Counter[str] = Counter()

    def _in_vocab(target: Any) -> bool:
        oov_here: set[str] = set()
        for residue in residue_set.tokenize(str(target)):
            try:
                residue_set.get_mass(residue)
            except KeyError:
                oov_here.add(residue)
        oov_counts.update(oov_here)  # count each offending residue once per target
        return not oov_here

    mask = targets.apply(_in_vocab)
    return mask, oov_counts


def _metric_block(df_subset: pd.DataFrame, residue_set: ResidueSet, metrics: Metrics) -> dict[str, Any]:
    """Compute the headline metrics over one subset of rows (empty subset → empty dict).

    Predictions are rewritten from the model's own notation to canonical UNIMOD (so the string-based
    AA error rate matches; the mass-based metrics are already remapping-aware via ``get_mass``).

    Args:
        df_subset: The rows to score (must have ``targets``, ``predictions``; ``log_probs`` optional).
        residue_set: The scoring residue set (with remapping + any sentinel masses).
        metrics: A :class:`Metrics` built on ``residue_set``.

    Returns:
        Dict of peptide/AA precision & recall, AA error rate, and (if ``log_probs`` present) AUC and
        recall at 5% FDR. Empty if ``df_subset`` is empty.
    """
    if len(df_subset) == 0:
        return {}
    targets = df_subset["targets"].tolist()
    predictions = [_canonicalize_peptide(str(pep), residue_set) for pep in df_subset["predictions"].tolist()]
    aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(targets, predictions)
    block: dict[str, Any] = {
        "peptide_recall": pep_recall,
        "peptide_precision": pep_prec,
        "aa_recall": aa_recall,
        "aa_precision": aa_prec,
        "aa_error_rate": metrics.compute_aa_er(targets, predictions),
    }
    if "log_probs" in df_subset.columns:
        # Empty/unpredictable rows have NaN log-probs; treat as lowest confidence so they rank last.
        confidence = np.exp(df_subset["log_probs"].astype(float)).fillna(0.0)
        try:
            block["auc"] = metrics.calc_auc(targets, predictions, confidence)
            recall_at_fdr, _threshold = metrics.find_recall_at_fdr(targets, predictions, confidence, fdr=0.05)
            block["recall_at_0.05_fdr"] = recall_at_fdr
        except Exception as exc:  # confidence-curve metrics are non-essential; keep the headline metrics
            logger.warning(f"AUC / recall@FDR skipped ({exc})")
    return block


def score_predictions(  # TODO if true==filtered don't need to compute twice
    csv_path: str,
    output_dir: Optional[str] = None,
    *,
    residue_config: Optional[str] = None,
    residue_remapping: Optional[dict[str, str]] = None,
    per_group: bool = False,
    group_column: str = "group",
) -> dict[str, Any]:
    """Score a prediction CSV against **two denominators**, with the shared ``Metrics`` + ``ResidueSet``.

    Every row is scored twice:

    * **true** (headline, unsuffixed keys) — over *all* labelled rows in the CSV. Rows the model's own
      preprocessing discarded have empty predictions and count as misses. This is the honest
      "coverage on the spectra it accepts" denominator. (Rows the model structurally cannot handle —
      invalid charge, or targets using residues it cannot emit — are dropped upstream at prediction and
      never reach the CSV; see :func:`filter_unpredictable_rows`.)
    * **filtered** (``*_predictable`` keys) — over ``predictable == True`` rows only, i.e. how well the
      model does on the spectra it can actually attempt.

    Only rows with no target (de novo / unlabelled) are excluded from both. Targets whose residues are
    absent from the scoring residue set are made scorable via a sentinel mass (guaranteed misses) so
    they still count toward the true denominator rather than crashing.

    Predictions are written in each model's own residue notation; ``residue_remapping`` (applied
    per-token by the ``ResidueSet``) rewrites them to canonical UNIMOD masses. Pass via :data:`MODEL_SCORING`.

    Args:
        csv_path: Path to a prediction CSV in the canonical schema.
        output_dir: Directory for the ``summary.json`` (created if given).
        residue_config: Path to the residue-masses YAML.
        residue_remapping: Model-notation → UNIMOD token remapping for the scoring residue set.
        per_group: If True, also compute a per-group metric block for each distinct value of
            ``group_column`` (e.g. one dataset/species per group in a combined multi-species run),
            using the same residue set + sentinels as the overall block.
        group_column: The column to group by when ``per_group`` is set (default ``group``).

    Returns:
        ``{"overall": {...}}`` — true-denominator metrics (unsuffixed) + filtered-denominator metrics
        (``*_predictable``) + coverage counts. When ``per_group`` is set, also ``{"groups": {name:
        {...}}}`` with the same block shape per group. Also written to ``summary.json``.
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)
    # Empty predictions (a real outcome: e.g. all beams below min_peptide_len, an all-blank CTC path,
    # or a row the predictor dropped as unpredictable) are blank cells read back as NaN floats.
    # Normalise them to "" so the scoring stack only sees strings — "" tokenises to [] and is a miss.
    if "predictions" in df.columns:
        df["predictions"] = df["predictions"].fillna("")
    if "targets" in df.columns:
        df = df[df["targets"].notna() & (df["targets"].astype(str).str.len() > 0)]
    df = df.reset_index(drop=True)
    n_true = len(df)
    if n_true == 0:
        logger.warning("No labelled predictions to score.")
        return {"overall": {"n": 0}}

    predictable_mask = df["predictable"].astype(str).str.lower().isin(["true", "1"]) if "predictable" in df.columns else pd.Series([True] * n_true)
    n_predictable = int(predictable_mask.sum())

    # Target residues absent from the scoring residue set can't be mass-scored. To count such targets
    # as misses (true denominator) rather than dropping them, register each with a unique sentinel mass:
    # the matcher then runs, and since no model emits these residues, the peptide is a guaranteed miss.
    residue_set = _build_residue_set(residue_config, residue_remapping=residue_remapping)
    in_scoring_vocab, unscorable_counts = _targets_in_vocab(df["targets"], residue_set)  # TODO check why not caughthere
    n_target_unscorable = int((~in_scoring_vocab).sum())
    if unscorable_counts:
        sentinels = {residue: _SENTINEL_MASS + i for i, residue in enumerate(unscorable_counts)}
        residue_set = _build_residue_set(residue_config, residue_remapping=residue_remapping, extra_masses=sentinels)

    metrics = Metrics(residue_set=residue_set)
    overall: dict[str, Any] = {"n": n_true}
    overall.update(_metric_block(df, residue_set, metrics))  # true denominator (all labelled rows) # TODO slow
    overall["n_predictable"] = n_predictable
    overall["n_unpredictable"] = n_true - n_predictable
    overall["n_target_unscorable"] = n_target_unscorable
    for key, value in _metric_block(df[predictable_mask].reset_index(drop=True), residue_set, metrics).items():
        overall[f"{key}_predictable"] = value  # filtered denominator (predictable rows only)

    results: dict[str, Any] = {"overall": overall}

    if per_group and group_column in df.columns:
        groups: dict[str, Any] = {}
        for group in df[group_column].astype(str).unique():
            if group in ("", "no_group", "nan", "None"):
                continue
            sub = df[df[group_column].astype(str) == group].reset_index(drop=True)
            g_predictable = (
                sub["predictable"].astype(str).str.lower().isin(["true", "1"]) if "predictable" in sub.columns else pd.Series([True] * len(sub))
            )
            block: dict[str, Any] = {"n": len(sub)}
            block.update(_metric_block(sub, residue_set, metrics))  # true denominator
            block["n_predictable"] = int(g_predictable.sum())
            block["n_unpredictable"] = len(sub) - int(g_predictable.sum())
            for key, value in _metric_block(sub[g_predictable].reset_index(drop=True), residue_set, metrics).items():
                block[f"{key}_predictable"] = value  # filtered denominator
            groups[group] = block
        results["groups"] = groups
        logger.info(f"Per-group metrics computed for {len(groups)} group(s): {sorted(groups)}")

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote summary to {out_dir / 'summary.json'}")
    logger.info(
        f"Scored {n_total:,} rows: true denominator n={n_true:,} "
        f"({n_true - n_predictable:,} unpredictable counted as misses), filtered denominator n={n_predictable:,}"
    )
    true_recall = overall.get("peptide_recall", float("nan"))
    true_aer = overall.get("aa_error_rate", float("nan"))
    logger.info(f"  true:      peptide_recall={true_recall:.4f}  aa_error_rate={true_aer:.4f}")
    if "peptide_recall_predictable" in overall:
        logger.info(
            f"  filtered:  peptide_recall={overall['peptide_recall_predictable']:.4f}  aa_error_rate={overall['aa_error_rate_predictable']:.4f}"
        )
    return results
