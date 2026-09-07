r"""Verify if peptide_calc_mz matches our calculation from the sequence column.

PURPOSE:
========
This script checks whether the `peptide_calc_mz` column in the data matches
what we calculate from the `sequence` column. It tests both:
1. The sequence as-is
2. The sequence with carbamidomethylated cysteines (C -> C[UNIMOD:4])

Optional search-data mode (``--search-data``) additionally checks whether TMT or iTRAQ
lysine labeling is implicit in ``peptide_calc_mz`` but missing from the sequence string:
bare ``K`` → ``K[UNIMOD:737]``, ``K[UNIMOD:2016]``, or ``K[UNIMOD:214]`` after carb on C,
using the same plain-AA + bare-K subset as the cysteine story. **DIA** files are excluded
from TMT/iTRAQ checks (mixed projects). It is a **fatal error** if a non-DIA row has
``quant`` indicating TMT and ``modifications`` containing iTRAQ. N-terminal-only isobaric
labels (no bare K in the sequence column) are not detected by this K-only rule.

This helps identify whether:
- The labeling software calculated mass correctly from its internal sequence
- The sequence column is missing carbamidomethylation that was used in calc_mz
- TMT/iTRAQ on lysine is implicit in calc_mz but not written on K
- There are other discrepancies between sequence and calculated mass

OUTPUT COLUMNS:
===============
- project: Project identifier
- total_rows: Total rows in scope (before unknown-token skips). Rows with null or non-positive
  ``precursor_charge`` are excluded (typical DIA); they are not scored against ``peptide_calc_mz``.
- rows_with_unmodified_cysteine: Rows with bare C (candidates for implicit carb in calc_mz)
- calc_mz_matches_as_is / calc_mz_match_rate_as_is_pct: Match vs sequence as written (denominator: processed rows)
- calc_mz_matches_after_carb_all_rows / calc_mz_match_rate_after_carb_pct: Match vs sequence after
  applying carbamidomethylation to unmodified C on every row; unchanged if no bare C (same denominator)
- calc_mz_matches_after_carb_unmod_c_rows / calc_mz_match_rate_after_carb_unmod_c_rows_pct: After-carb match
  count and rate among rows with bare C **and** a plain-AA sequence only (no [mod] / non-letter symbols), so
  other modification label noise does not dilute the C/carb signal (denominator: rows_entirely_unmodified_seq_with_bare_c)
- avg_error_*_ppm: Mean PPM vs peptide_calc_mz for as-is, after-carb (all rows), after-carb (unmod C rows only)
- suggest_explicit_carbamidomethylation_project: True when the same gate as
  ``apply_carbamido_from_calc_mz_report.select_projects_for_carb`` would select this project
- TMT / iTRAQ columns (null when ``--search-data`` not passed): per-project aggregates and flags for
  implicit lysine labeling vs sequence. Plain-seq bare-K subset: as-is, label-only (no carb on C),
  carb + label (see column names in CSV)

READING THE RATES (1) = as-is %, (2) = after-carb over all processed, (3) = after-carb % among plain-AA rows with bare C only:
===============================================================================================================
- (1) = 100%: Sequence string and peptide_calc_mz align; nothing to fix for this check.
- (2) > (1): Some bare-C rows match calc_mz only after implicit carbamidomethylation — typical when the table
  omits C[UNIMOD:4] but the search used alkylated mass.
- If (3) = 100% but (2) < 100%: Every bare-C **plain-sequence** row is explained by carb; rows that still fail (2) have no bare C
  (or C is already annotated), so the carb transform does not change their mass. Remaining gaps point to
  other peptide-label or mass-calculation mismatches vs peptide_calc_mz (other mods, charge, tokenization,
  sequence vs what the pipeline used, etc.) — not missing C alkylation on the written sequence.
- If (3) < 100%: Some plain-sequence bare-C rows still disagree after carb — look beyond “add carb on C”
  (wrong mod dictionary, etc.) for that subset.

USAGE:
======
python verify_calc_mz.py \
    --input-dir <data-root>/lcfm/ \
    --output-csv calc_mz_verification.csv \
    --tolerance 10 \
    --verbose

python verify_calc_mz.py \
    --input-dir <data-root>/lcfm/ \
    --output-csv calc_mz_verification.csv \
    --search-data search_data_with_new_projects.xlsx \
    --tmt-projects-yaml bad_tmt_projects.yaml \
    --lysine-label-file-csv lysine_label_files.csv
"""

import polars as pl
import os
import re
import yaml
import logging
import time
from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Set, Tuple
import typer

from instanovo.utils.residues import ResidueSet, H2O_MASS, PROTON_MASS_AMU


app = typer.Typer()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level constants for CLI options
INPUT_DIR_OPTION = typer.Option(
    "<data-root>/lcfm/",
    "--input-dir",
    "-i",
    help="Input directory containing parquet files organized by project subfolders",
)
RESIDUE_MASSES_FILE_OPTION = typer.Option(
    "mod_dicts/residue_masses.yaml",
    "--residue-masses-file",
    "-r",
    help="Path to residue masses file",
)
TOLERANCE_OPTION = typer.Option(
    10,
    "--tolerance",
    "-t",
    help="PPM tolerance for m/z matching",
)
OUTPUT_CSV_OPTION = typer.Option(
    ...,
    "--output-csv",
    "-o",
    help="Path to write the output CSV report",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose logging",
)
SEARCH_DATA_OPTION = typer.Option(
    None,
    "--search-data",
    help="Search data Excel with project, file path, acquisition, quant, modifications",
)
TMT_PROJECTS_YAML_OPTION = typer.Option(
    "bad_tmt_projects.yaml",
    "--tmt-projects-yaml",
    help="YAML mapping TMTplex groups to projects (tmt_6_8_10 / tmt_16_18)",
)
LYSINE_LABEL_FILE_CSV_OPTION = typer.Option(
    None,
    "--lysine-label-file-csv",
    help="Optional per-file TMT/iTRAQ lysine report (requires --search-data)",
)


def extract_file_name(path_str: str) -> str:
    """Filename stem; matches search-data / parquet basename logic used elsewhere."""
    path_obj = PureWindowsPath(path_str)
    name = path_obj.name
    compound_suffixes = [
        ".mzML.ipc",
        ".mzml.ipc",
        ".mzML.gz",
        ".mzml.gz",
        ".mzml.parquet",
        ".mzML.parquet",
    ]
    for suffix in compound_suffixes:
        if name.lower().endswith(suffix.lower()):
            return name[: -len(suffix)]
    return Path(name).stem


def load_residue_masses(residue_masses_file: str) -> dict[str, float]:
    """Load residue masses from a YAML file."""
    with open(residue_masses_file, "r") as f:
        data = yaml.safe_load(f)
    residues: dict[str, float] = data.get("residues", data)
    return residues


def create_residue_set(residue_masses_file: str) -> ResidueSet:
    """Create a ResidueSet with appropriate tokenizer regex."""
    residue_masses = load_residue_masses(residue_masses_file)
    residue_set = ResidueSet(residue_masses=residue_masses)
    residue_set.tokenizer_regex = (
        r"(\[[^\]]+\]"
        r"|\([^)]+\)"
        r"|[+-]?\d+(?:\.\d+)?"
        r"|[+-]?\.\d+"
        r")|"
        r"([A-Z]"
        r"(?:\[[^\]]+\]"
        r"|\([^)]+\)"
        r"|[+-]?\d+(?:\.\d+)?"
        r"|[+-]?\.\d+)?"
        r")"
    )
    return residue_set


def carbamidomethylate_cysteines(sequence: str) -> str:
    """Replace all unmodified C with C[UNIMOD:4]."""
    return re.sub(r"C(?!\[)", "C[UNIMOD:4]", sequence)


def label_unmodified_lysines(sequence: str, unimod_id: str) -> str:
    """Replace every unmodified K with K[UNIMOD:<id>]."""
    return re.sub(r"K(?!\[)", f"K[UNIMOD:{unimod_id}]", sequence)


def is_tmt_quant(quant: object) -> bool:
    """True if search-data quant value indicates TMT."""
    if quant is None:
        return False
    return str(quant).strip().casefold() == "tmt"


def modifications_contains_itraq(modifications: object) -> bool:
    """True if modifications string mentions iTRAQ (substring, case-insensitive)."""
    if modifications is None:
        return False
    return "itraq" in str(modifications).casefold()


def has_bare_lysine(sequence: str) -> bool:
    """True if sequence has lysine not already written as K[...]."""
    return "K" in sequence and not re.search(r"K\[", sequence)


def load_tmt_unimod_by_project(tmt_projects_yaml: str) -> Dict[str, str]:
    """Map project id -> UNIMOD id string for TMT on K (737 or 2016)."""
    with open(tmt_projects_yaml, "r") as f:
        data = yaml.safe_load(f)
    out: Dict[str, str] = {}
    for proj in data.get("tmt_6_8_10", []) or []:
        p = str(proj).strip()
        if p in out:
            raise ValueError(
                f"Duplicate project {p!r} in {tmt_projects_yaml} (tmt_6_8_10 / tmt_16_18)"
            )
        out[p] = "737"
    for proj in data.get("tmt_16_18", []) or []:
        p = str(proj).strip()
        if p in out:
            raise ValueError(
                f"Project {p!r} appears in both tmt_6_8_10 and tmt_16_18 in {tmt_projects_yaml}"
            )
        out[p] = "2016"
    return out


def assert_no_tmt_quant_and_itraq_modifications_conflict(df: pl.DataFrame) -> None:
    """Raise ValueError if any non-DIA row has TMT quant and iTRAQ in modifications."""
    bad: List[str] = []
    for row in df.iter_rows(named=True):
        acq = row.get("acquisition")
        if acq is not None and str(acq).strip() == "DIA":
            continue
        quant = row.get("quant")
        mods = row.get("modifications")
        if is_tmt_quant(quant) and modifications_contains_itraq(mods):
            project = row.get("project")
            fp = row.get("file path")
            fname = extract_file_name(str(fp)) if fp is not None else "?"
            bad.append(f"  {project}/{fname}: quant={quant!r} modifications={mods!r}")
    if bad:
        raise ValueError(
            "Non-DIA rows must not combine TMT quant with iTRAQ in modifications:\n"
            + "\n".join(bad)
        )


def _require_search_excel_columns(df: pl.DataFrame) -> None:
    required = ["project", "file path", "acquisition", "quant", "modifications"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Search data missing columns {missing}. Found columns: {df.columns}"
        )


def _accumulate_tmt_itraq_from_search_rows(
    df: pl.DataFrame, project_to_unimod: Dict[str, str]
) -> Tuple[Dict[Tuple[str, str], str], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    tmt_files: Dict[Tuple[str, str], str] = {}
    itraq_files: Set[Tuple[str, str]] = set()
    tmt_quant_missing_yaml: Set[Tuple[str, str]] = set()

    for row in df.iter_rows(named=True):
        acq = row["acquisition"]
        if acq is None or str(acq).strip() == "DIA":
            continue
        fp = row["file path"]
        if fp is None:
            continue
        project = str(row["project"]).strip()
        filename = extract_file_name(str(fp))
        key = (project, filename)
        quant = row["quant"]
        modifications = row["modifications"]

        if modifications_contains_itraq(modifications):
            itraq_files.add(key)
        if not is_tmt_quant(quant):
            continue
        if project in project_to_unimod:
            tmt_files[key] = project_to_unimod[project]
        else:
            tmt_quant_missing_yaml.add(key)

    return tmt_files, itraq_files, tmt_quant_missing_yaml


def _raise_if_tmt_itraq_file_overlap(
    tmt_files: Dict[Tuple[str, str], str], itraq_files: Set[Tuple[str, str]]
) -> None:
    overlap = set(tmt_files.keys()) & itraq_files
    if not overlap:
        return
    examples = ", ".join(f"{p}/{f}" for p, f in sorted(overlap)[:10])
    raise ValueError(
        f"Same file appears as both TMT (yaml) and iTRAQ in merged search data: {examples}"
    )


def load_search_data_lysine_maps(
    search_data_path: str,
    tmt_projects_yaml: str,
) -> Tuple[Dict[Tuple[str, str], str], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Build TMT file→UNIMOD map, iTRAQ file set, and TMT-quant files missing from YAML.

    Returns:
        tmt_files: (project, filename) -> unimod id for TMT quant + non-DIA + project in YAML
        itraq_files: (project, filename) with iTRAQ in modifications + non-DIA
        tmt_quant_missing_yaml: TMT quant + non-DIA but project not listed in YAML (warn when scanning)
    """
    df = pl.read_excel(search_data_path)
    _require_search_excel_columns(df)
    assert_no_tmt_quant_and_itraq_modifications_conflict(df)
    project_to_unimod = load_tmt_unimod_by_project(tmt_projects_yaml)
    tmt_files, itraq_files, tmt_quant_missing_yaml = (
        _accumulate_tmt_itraq_from_search_rows(df, project_to_unimod)
    )
    _raise_if_tmt_itraq_file_overlap(tmt_files, itraq_files)
    return tmt_files, itraq_files, tmt_quant_missing_yaml


def is_entirely_unmodified_sequence(sequence: str) -> bool:
    """True if sequence is plain one-letter amino acids only (no mods in the string)."""
    return bool(sequence and re.fullmatch(r"[A-Z]+", sequence))


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    return str(timedelta(seconds=int(seconds)))


def calculate_mz(
    sequence: str, charge: int, residue_set: ResidueSet
) -> Optional[float]:
    """Calculate m/z from a sequence and charge."""
    try:
        tokens = residue_set.tokenize(sequence)
        total_mass = 0.0
        for token in tokens:
            lookup_token = token
            if residue_set.residue_remapping and token in residue_set.residue_remapping:
                lookup_token = residue_set.residue_remapping[token]
            if lookup_token in residue_set.residue_masses:
                total_mass += residue_set.residue_masses[lookup_token]
            else:
                return None  # Unknown token
        total_mass += H2O_MASS
        if charge > 0:
            mz = (total_mass / charge) + PROTON_MASS_AMU
        else:
            mz = total_mass
        return float(mz)
    except Exception:
        return None


def calculate_ppm_error(calc_mz: float, reference_mz: float) -> float:
    """Calculate PPM error between two m/z values."""
    if reference_mz == 0:
        return float("inf")
    return abs(calc_mz - reference_mz) / reference_mz * 1e6


def find_project_folders(input_dir: str) -> List[str]:
    """Find all project subfolders containing parquet files."""
    projects = []
    for entry in os.listdir(input_dir):
        project_path = os.path.join(input_dir, entry)
        if os.path.isdir(project_path):
            parquet_files = [
                f for f in os.listdir(project_path) if f.endswith(".parquet")
            ]
            if parquet_files:
                projects.append(entry)
    return sorted(projects)


def find_parquet_files_in_project(input_dir: str, project: str) -> List[str]:
    """Find all parquet files in a project folder."""
    project_path = os.path.join(input_dir, project)
    parquet_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


@dataclass
class ProjectCalcMzStats:
    """Statistics for calc_mz verification per project."""

    project: str
    total_rows: int = 0
    rows_with_cysteine: int = 0
    rows_entirely_unmodified_seq_with_bare_c: int = 0
    calc_mz_matches_as_is: int = 0
    calc_mz_matches_after_carb_all_rows: int = 0
    calc_mz_matches_after_carb_unmod_c_rows: int = 0
    errors_as_is: List[float] = field(default_factory=list)
    errors_after_carb_all: List[float] = field(default_factory=list)
    errors_after_carb_unmod_c: List[float] = field(default_factory=list)
    skipped_unknown_tokens: int = 0


@dataclass
class LysineLabelFileAccumulator:
    """Counts for one parquet under TMT or iTRAQ lysine labeling check."""

    project: str
    filename: str
    lysine_label_kind: str
    unimod_k: str
    rows_plain_seq_bare_k: int = 0
    matches_as_is_plain_bare_k: int = 0
    matches_label_only_plain_bare_k: int = 0
    matches_after_carb_label_plain_bare_k: int = 0


def _process_row(
    row: Dict,
    residue_set: ResidueSet,
    tolerance: float,
    stats: ProjectCalcMzStats,
) -> None:
    """Process a single row and update stats."""
    sequence = row["sequence"]
    charge = row["precursor_charge"]
    peptide_calc_mz = row["peptide_calc_mz"]

    # Calculate m/z as-is
    our_mz_as_is = calculate_mz(sequence, charge, residue_set)
    if our_mz_as_is is None:
        stats.skipped_unknown_tokens += 1
        return

    error_as_is = calculate_ppm_error(our_mz_as_is, peptide_calc_mz)
    stats.errors_as_is.append(error_as_is)

    if error_as_is <= tolerance:
        stats.calc_mz_matches_as_is += 1

    # After carb: apply to every row (no-op when there is no unmodified C)
    seq_after_carb = carbamidomethylate_cysteines(sequence)
    our_mz_after_carb = calculate_mz(seq_after_carb, charge, residue_set)
    if our_mz_after_carb is not None:
        error_after_carb = calculate_ppm_error(our_mz_after_carb, peptide_calc_mz)
        stats.errors_after_carb_all.append(error_after_carb)
        if error_after_carb <= tolerance:
            stats.calc_mz_matches_after_carb_all_rows += 1

    has_bare_c = "C" in sequence and not re.search(r"C\[", sequence)
    if has_bare_c:
        stats.rows_with_cysteine += 1
    if has_bare_c and is_entirely_unmodified_sequence(sequence):
        stats.rows_entirely_unmodified_seq_with_bare_c += 1
        if our_mz_after_carb is not None:
            stats.errors_after_carb_unmod_c.append(error_after_carb)
            if error_after_carb <= tolerance:
                stats.calc_mz_matches_after_carb_unmod_c_rows += 1


def _process_row_lysine_plain_bare_k(
    row: Dict,
    residue_set: ResidueSet,
    tolerance: float,
    unimod_id: str,
    lys_acc: LysineLabelFileAccumulator,
) -> None:
    """Update lysine accumulator for plain-AA rows with bare K."""
    sequence = row["sequence"]
    charge = row["precursor_charge"]
    peptide_calc_mz = row["peptide_calc_mz"]
    if not is_entirely_unmodified_sequence(sequence) or not has_bare_lysine(sequence):
        return

    our_mz_as_is = calculate_mz(sequence, charge, residue_set)
    if our_mz_as_is is None:
        return

    lys_acc.rows_plain_seq_bare_k += 1
    if calculate_ppm_error(our_mz_as_is, peptide_calc_mz) <= tolerance:
        lys_acc.matches_as_is_plain_bare_k += 1

    seq_label_only = label_unmodified_lysines(sequence, unimod_id)
    our_label_only = calculate_mz(seq_label_only, charge, residue_set)
    if (
        our_label_only is not None
        and calculate_ppm_error(our_label_only, peptide_calc_mz) <= tolerance
    ):
        lys_acc.matches_label_only_plain_bare_k += 1

    seq_labeled = label_unmodified_lysines(
        carbamidomethylate_cysteines(sequence), unimod_id
    )
    our_after = calculate_mz(seq_labeled, charge, residue_set)
    if (
        our_after is not None
        and calculate_ppm_error(our_after, peptide_calc_mz) <= tolerance
    ):
        lys_acc.matches_after_carb_label_plain_bare_k += 1


def _process_file(
    file_path: str,
    residue_set: ResidueSet,
    tolerance: float,
    stats: ProjectCalcMzStats,
    lysine_acc: Optional[LysineLabelFileAccumulator] = None,
) -> None:
    """Process a single parquet file."""
    schema = pl.scan_parquet(file_path).collect_schema()
    required_cols = ["sequence", "precursor_charge", "peptide_calc_mz"]

    if not all(col in schema for col in required_cols):
        logger.debug(f"Skipping {file_path}: missing required columns")
        return

    df = (
        pl.scan_parquet(file_path)
        .select(required_cols)
        .filter(
            pl.col("sequence").is_not_null()
            & pl.col("precursor_charge").is_not_null()
            & pl.col("peptide_calc_mz").is_not_null()
            & (pl.col("precursor_charge") > 0)
            & ~pl.col("sequence").str.contains("IN:", literal=True)
        )
        .collect()
    )

    stats.total_rows += len(df)

    for row in df.iter_rows(named=True):
        try:
            _process_row(row, residue_set, tolerance, stats)
            if lysine_acc is not None:
                _process_row_lysine_plain_bare_k(
                    row, residue_set, tolerance, lysine_acc.unimod_k, lysine_acc
                )
        except Exception as e:
            logger.debug(f"Error processing row: {e}")


def _finalize_lysine_file_row(acc: LysineLabelFileAccumulator) -> dict:
    """Build CSV dict from accumulator; empty files still get a row when analyzed."""
    d = acc.rows_plain_seq_bare_k
    rate_as_is = (acc.matches_as_is_plain_bare_k / d * 100) if d > 0 else 0.0
    rate_label_only = (acc.matches_label_only_plain_bare_k / d * 100) if d > 0 else 0.0
    rate_after = (acc.matches_after_carb_label_plain_bare_k / d * 100) if d > 0 else 0.0
    suggest = (
        d > 0
        and acc.matches_after_carb_label_plain_bare_k == d
        and acc.matches_as_is_plain_bare_k < d
    )
    return {
        "project": acc.project,
        "filename": acc.filename,
        "lysine_label_kind": acc.lysine_label_kind,
        "unimod_k": acc.unimod_k,
        "rows_plain_seq_bare_k": acc.rows_plain_seq_bare_k,
        "calc_mz_matches_as_is_plain_bare_k_rows": acc.matches_as_is_plain_bare_k,
        "calc_mz_matches_label_only_plain_bare_k_rows": (
            acc.matches_label_only_plain_bare_k
        ),
        "calc_mz_matches_after_carb_label_plain_bare_k_rows": (
            acc.matches_after_carb_label_plain_bare_k
        ),
        "match_rate_as_is_plain_bare_k_pct": round(rate_as_is, 2),
        "match_rate_label_only_plain_bare_k_pct": round(rate_label_only, 2),
        "match_rate_after_carb_label_plain_bare_k_pct": round(rate_after, 2),
        "suggest_explicit_lysine_labeling": suggest,
    }


def analyze_project(
    input_dir: str,
    project: str,
    residue_set: ResidueSet,
    tolerance: float,
    tmt_files: Optional[Dict[Tuple[str, str], str]] = None,
    itraq_files: Optional[Set[Tuple[str, str]]] = None,
    tmt_quant_missing_yaml: Optional[Set[Tuple[str, str]]] = None,
    lysine_file_rows_out: Optional[List[dict]] = None,
) -> ProjectCalcMzStats:
    """Analyze calc_mz for a project."""
    parquet_files = find_parquet_files_in_project(input_dir, project)

    stats = ProjectCalcMzStats(project=project)
    tmt_files = tmt_files or {}
    itraq_files = itraq_files or set()
    tmt_quant_missing_yaml = tmt_quant_missing_yaml or set()

    for file_path in parquet_files:
        filename = extract_file_name(file_path)
        key = (project, filename)
        lysine_acc: Optional[LysineLabelFileAccumulator] = None

        if key in itraq_files:
            lysine_acc = LysineLabelFileAccumulator(
                project=project,
                filename=filename,
                lysine_label_kind="iTRAQ",
                unimod_k="214",
            )
        elif key in tmt_files:
            lysine_acc = LysineLabelFileAccumulator(
                project=project,
                filename=filename,
                lysine_label_kind="TMT",
                unimod_k=tmt_files[key],
            )
        elif key in tmt_quant_missing_yaml:
            logger.warning(
                "TMT quant in search data but project not in TMT YAML — skipping "
                "lysine check for %s/%s",
                project,
                filename,
            )

        try:
            _process_file(file_path, residue_set, tolerance, stats, lysine_acc)
            if lysine_acc is not None and lysine_file_rows_out is not None:
                lysine_file_rows_out.append(_finalize_lysine_file_row(lysine_acc))
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")

    return stats


def _suggest_explicit_carbamidomethylation_project(stats: ProjectCalcMzStats) -> bool:
    """Same logical gate as apply_carbamido_from_calc_mz_report.select_projects_for_carb."""
    total = stats.total_rows - stats.skipped_unknown_tokens
    if total <= 0:
        return False
    if stats.calc_mz_matches_as_is >= total:
        return False
    if stats.rows_entirely_unmodified_seq_with_bare_c <= 0:
        return False
    if stats.calc_mz_matches_after_carb_unmod_c_rows <= 0:
        return False
    return (
        stats.calc_mz_matches_after_carb_unmod_c_rows
        == stats.rows_entirely_unmodified_seq_with_bare_c
    )


def _lysine_project_columns(project: str, lysine_file_rows: List[dict]) -> dict:
    """Aggregate per-file lysine rows into project-level TMT and iTRAQ columns."""

    def agg_for_kind(rows_subset: List[dict], prefix: str) -> dict:
        n_files = len(rows_subset)
        plain = sum(int(r["rows_plain_seq_bare_k"]) for r in rows_subset)
        m_as_is = sum(
            int(r["calc_mz_matches_as_is_plain_bare_k_rows"]) for r in rows_subset
        )
        m_label_only = sum(
            int(r["calc_mz_matches_label_only_plain_bare_k_rows"]) for r in rows_subset
        )
        m_after = sum(
            int(r["calc_mz_matches_after_carb_label_plain_bare_k_rows"])
            for r in rows_subset
        )
        rate_as_is = (m_as_is / plain * 100) if plain > 0 else 0.0
        rate_label_only = (m_label_only / plain * 100) if plain > 0 else 0.0
        rate_after = (m_after / plain * 100) if plain > 0 else 0.0
        any_suggest = any(
            bool(r["suggest_explicit_lysine_labeling"]) for r in rows_subset
        )
        return {
            f"{prefix}_eligible_files_analyzed": n_files,
            f"{prefix}_rows_plain_seq_bare_k": plain,
            f"{prefix}_calc_mz_matches_as_is_plain_bare_k_rows": m_as_is,
            f"{prefix}_calc_mz_matches_label_only_plain_bare_k_rows": m_label_only,
            f"{prefix}_calc_mz_matches_after_carb_label_plain_bare_k_rows": m_after,
            f"{prefix}_match_rate_as_is_plain_bare_k_pct": round(rate_as_is, 2),
            f"{prefix}_match_rate_label_only_plain_bare_k_pct": round(
                rate_label_only, 2
            ),
            f"{prefix}_match_rate_after_carb_label_plain_bare_k_pct": round(
                rate_after, 2
            ),
            f"project_has_{prefix}_file_needing_k_label": any_suggest,
        }

    rows = [r for r in lysine_file_rows if r["project"] == project]
    tmt_rows = [r for r in rows if r["lysine_label_kind"] == "TMT"]
    itraq_rows = [r for r in rows if r["lysine_label_kind"] == "iTRAQ"]
    return {**agg_for_kind(tmt_rows, "tmt"), **agg_for_kind(itraq_rows, "itraq")}


def _null_lysine_project_columns() -> dict:
    """Placeholder columns when --search-data is not used."""
    return {k: None for k in _lysine_project_columns("_", []).keys()}


def _create_output_row(
    stats: ProjectCalcMzStats,
    lysine_project_extras: Optional[dict] = None,
) -> dict:
    """Create output row from project stats."""
    total = stats.total_rows - stats.skipped_unknown_tokens
    match_rate_as_is = (stats.calc_mz_matches_as_is / total * 100) if total > 0 else 0.0
    match_rate_after_carb_total = (
        (stats.calc_mz_matches_after_carb_all_rows / total * 100) if total > 0 else 0.0
    )
    match_rate_after_carb_unmod_c = (
        (
            stats.calc_mz_matches_after_carb_unmod_c_rows
            / stats.rows_entirely_unmodified_seq_with_bare_c
            * 100
        )
        if stats.rows_entirely_unmodified_seq_with_bare_c > 0
        else 0.0
    )
    avg_error_as_is = (
        sum(stats.errors_as_is) / len(stats.errors_as_is) if stats.errors_as_is else 0.0
    )
    avg_error_after_all = (
        sum(stats.errors_after_carb_all) / len(stats.errors_after_carb_all)
        if stats.errors_after_carb_all
        else 0.0
    )
    avg_error_after_unmod_c = (
        sum(stats.errors_after_carb_unmod_c) / len(stats.errors_after_carb_unmod_c)
        if stats.errors_after_carb_unmod_c
        else 0.0
    )

    row = {
        "project": stats.project,
        "total_rows": stats.total_rows,
        "rows_with_unmodified_cysteine": stats.rows_with_cysteine,
        "skipped_unknown_tokens": stats.skipped_unknown_tokens,
        "calc_mz_matches_as_is": stats.calc_mz_matches_as_is,
        "calc_mz_matches_after_carb_all_rows": stats.calc_mz_matches_after_carb_all_rows,
        "calc_mz_matches_after_carb_unmod_c_rows": stats.calc_mz_matches_after_carb_unmod_c_rows,
        "calc_mz_match_rate_as_is_pct": round(match_rate_as_is, 2),
        "calc_mz_match_rate_after_carb_pct": round(match_rate_after_carb_total, 2),
        "calc_mz_match_rate_after_carb_unmod_c_rows_pct": round(
            match_rate_after_carb_unmod_c, 2
        ),
        "avg_error_vs_calc_mz_as_is_ppm": round(avg_error_as_is, 2),
        "avg_error_vs_calc_mz_after_carb_ppm": round(avg_error_after_all, 2),
        "avg_error_vs_calc_mz_after_carb_unmod_c_rows_ppm": round(
            avg_error_after_unmod_c, 2
        ),
        "suggest_explicit_carbamidomethylation_project": (
            _suggest_explicit_carbamidomethylation_project(stats)
        ),
    }
    if lysine_project_extras is not None:
        row.update(lysine_project_extras)
    else:
        row.update(_null_lysine_project_columns())
    return row


def _log_summary(results: List[ProjectCalcMzStats]) -> None:
    """Log summary of results."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_rows = sum(r.total_rows for r in results)
    total_matches_as_is = sum(r.calc_mz_matches_as_is for r in results)
    total_matches_after_all = sum(
        r.calc_mz_matches_after_carb_all_rows for r in results
    )
    total_matches_after_unmod_c = sum(
        r.calc_mz_matches_after_carb_unmod_c_rows for r in results
    )
    total_with_c = sum(r.rows_with_cysteine for r in results)
    total_plain_bare_c = sum(
        r.rows_entirely_unmodified_seq_with_bare_c for r in results
    )
    total_skipped = sum(r.skipped_unknown_tokens for r in results)

    processed = total_rows - total_skipped
    match_rate_as_is = (total_matches_as_is / processed * 100) if processed > 0 else 0.0
    match_rate_after_carb_total = (
        (total_matches_after_all / processed * 100) if processed > 0 else 0.0
    )
    match_rate_after_carb_unmod_c = (
        (total_matches_after_unmod_c / total_plain_bare_c * 100)
        if total_plain_bare_c > 0
        else 0.0
    )

    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Skipped (unknown tokens): {total_skipped:,}")
    logger.info(f"Processed: {processed:,}")
    logger.info("")
    logger.info(
        f"Our calc matches peptide_calc_mz (as-is, / processed): "
        f"{total_matches_as_is:,} ({match_rate_as_is:.2f}%)"
    )
    logger.info(
        f"Our calc matches after carb on all rows (/ processed): "
        f"{total_matches_after_all:,} ({match_rate_after_carb_total:.2f}%)"
    )
    logger.info(f"Rows with unmodified cysteine: {total_with_c:,}")
    logger.info(
        f"Plain-AA rows with bare C (denom for carb / bare-C rate): {total_plain_bare_c:,}"
    )
    logger.info(
        f"Our calc matches after carb (/ plain-AA + bare-C rows): "
        f"{total_matches_after_unmod_c:,} ({match_rate_after_carb_unmod_c:.2f}%)"
    )
    logger.info("=" * 60)


LYSINE_FILE_REPORT_COLS = [
    "project",
    "filename",
    "lysine_label_kind",
    "unimod_k",
    "rows_plain_seq_bare_k",
    "match_rate_as_is_plain_bare_k_pct",
    "match_rate_label_only_plain_bare_k_pct",
    "match_rate_after_carb_label_plain_bare_k_pct",
    "suggest_explicit_lysine_labeling",
]


def _load_optional_search_maps(
    search_data: Optional[str], tmt_projects_yaml: str
) -> Tuple[
    Optional[Dict[Tuple[str, str], str]],
    Optional[Set[Tuple[str, str]]],
    Optional[Set[Tuple[str, str]]],
]:
    if not search_data:
        return None, None, None
    logger.info("Loading search data: %s", search_data)
    logger.info("TMT project YAML: %s", tmt_projects_yaml)
    tmt_files, itraq_files, tmt_quant_missing_yaml = load_search_data_lysine_maps(
        search_data, tmt_projects_yaml
    )
    logger.info(
        "Search data: %d TMT (yaml) files, %d iTRAQ files",
        len(tmt_files),
        len(itraq_files),
    )
    return tmt_files, itraq_files, tmt_quant_missing_yaml


def _analyze_all_projects(
    input_dir: str,
    projects: List[str],
    residue_set: ResidueSet,
    tolerance: float,
    tmt_files: Optional[Dict[Tuple[str, str], str]],
    itraq_files: Optional[Set[Tuple[str, str]]],
    tmt_quant_missing_yaml: Optional[Set[Tuple[str, str]]],
    lysine_file_rows: List[dict],
    search_data: Optional[str],
) -> List[ProjectCalcMzStats]:
    results: List[ProjectCalcMzStats] = []
    start_time = time.time()
    lysine_out = lysine_file_rows if search_data else None
    for i, project in enumerate(projects):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Processed %d/%d projects (%s)",
                i + 1,
                len(projects),
                format_time(elapsed),
            )
        stats = analyze_project(
            input_dir,
            project,
            residue_set,
            tolerance,
            tmt_files=tmt_files,
            itraq_files=itraq_files,
            tmt_quant_missing_yaml=tmt_quant_missing_yaml,
            lysine_file_rows_out=lysine_out,
        )
        results.append(stats)
    return results


def _verification_output_rows(
    results: List[ProjectCalcMzStats],
    lysine_file_rows: List[dict],
    search_data: Optional[str],
) -> List[dict]:
    rows_out: List[dict] = []
    for r in results:
        extras = (
            _lysine_project_columns(r.project, lysine_file_rows)
            if search_data
            else None
        )
        rows_out.append(_create_output_row(r, lysine_project_extras=extras))
    return rows_out


def _write_lysine_label_file_csv(path: str, lysine_file_rows: List[dict]) -> None:
    if lysine_file_rows:
        pl.DataFrame(lysine_file_rows).select(LYSINE_FILE_REPORT_COLS).write_csv(path)
    else:
        pl.DataFrame({c: [] for c in LYSINE_FILE_REPORT_COLS}).write_csv(path)
    logger.info("Lysine per-file report written to: %s", path)


def run_verification(
    input_dir: str,
    residue_masses_file: str,
    tolerance: float,
    output_csv: str,
    verbose: bool = False,
    search_data: Optional[str] = None,
    tmt_projects_yaml: str = "bad_tmt_projects.yaml",
    lysine_label_file_csv: Optional[str] = None,
) -> None:
    """Run the calc_mz verification."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Tolerance: {tolerance} ppm")
    logger.info(f"Output CSV: {output_csv}")

    if lysine_label_file_csv and not search_data:
        raise ValueError("--lysine-label-file-csv requires --search-data")

    lysine_file_rows: List[dict] = []
    tmt_files, itraq_files, tmt_quant_missing_yaml = _load_optional_search_maps(
        search_data, tmt_projects_yaml
    )

    logger.info("Initializing residue set...")
    residue_set = create_residue_set(residue_masses_file)

    logger.info("Finding projects...")
    projects = find_project_folders(input_dir)
    logger.info(f"Found {len(projects)} projects")

    if not projects:
        logger.warning("No projects found!")
        return

    logger.info("Analyzing projects...")
    results = _analyze_all_projects(
        input_dir,
        projects,
        residue_set,
        tolerance,
        tmt_files,
        itraq_files,
        tmt_quant_missing_yaml,
        lysine_file_rows,
        search_data,
    )

    logger.info("Generating report...")
    output_rows = _verification_output_rows(results, lysine_file_rows, search_data)
    pl.DataFrame(output_rows).sort("project").write_csv(output_csv)
    logger.info(f"Results written to: {output_csv}")

    if lysine_label_file_csv:
        _write_lysine_label_file_csv(lysine_label_file_csv, lysine_file_rows)

    _log_summary(results)


@app.command()
def main(
    input_dir: str = INPUT_DIR_OPTION,
    residue_masses_file: str = RESIDUE_MASSES_FILE_OPTION,
    tolerance: float = TOLERANCE_OPTION,
    output_csv: str = OUTPUT_CSV_OPTION,
    verbose: bool = VERBOSE_OPTION,
    search_data: Optional[str] = SEARCH_DATA_OPTION,
    tmt_projects_yaml: str = TMT_PROJECTS_YAML_OPTION,
    lysine_label_file_csv: Optional[str] = LYSINE_LABEL_FILE_CSV_OPTION,
) -> None:
    """Verify if peptide_calc_mz matches our calculation from sequence."""
    run_verification(
        input_dir=input_dir,
        residue_masses_file=residue_masses_file,
        tolerance=tolerance,
        output_csv=output_csv,
        verbose=verbose,
        search_data=search_data,
        tmt_projects_yaml=tmt_projects_yaml,
        lysine_label_file_csv=lysine_label_file_csv,
    )


if __name__ == "__main__":
    app()
