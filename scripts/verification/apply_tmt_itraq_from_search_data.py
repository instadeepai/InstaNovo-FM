r"""Apply tandem mass tag (TMT) or iTRAQ lysine labels from search-data Excel.

Run after search metadata is known so unmodified lysines in parquet sequences
can be written with the UNIMOD accession used by the original search
(TMT 6/8/10-plex → 737, TMT 16/18-plex → 2016, iTRAQ 4-plex → 214).

For each user-specified (project, tag kind), unmodified lysines are labelled
with that UNIMOD accession.

**TMT** (primary): each row must match both ``quant`` = TMT and TMT multiplex
markers in ``modifications`` (see ``modifications_match_tag_kind``).

**iTRAQ** (primary): only ``modifications`` are used — the row must match
**iTRAQ 4-plex** text (``itraq`` + ``4`` + optional ``plex`` / spacing; see
``modifications_match_itraq_4plex``). Other iTRAQ multiplex sets are not matched.
If ``quant`` is not exactly ``iTRAQ`` (case-insensitive), a **warning** is logged
but labelling still proceeds for those rows.

If no row qualifies on the primary rules, the script prints a quantification
summary and may require ``--confirm-project-wide PROJECT`` or interactive
confirmation before a **quant-only** fallback (TMT, or ``quant`` = iTRAQ for
iTRAQ specs).

**YAML spec file:** must be a single mapping ``project_id: TAG_KIND``. Duplicate
project keys are invalid in YAML (the last wins). To run the same project with
more than one tag kind (for example mixed TMT 10-plex and 16-plex files), use
repeated ``--spec PROJECT:TAG`` on the command line and/or split entries across
multiple YAML files.

CLI::

    uv run python -m scripts.verification.apply_tmt_itraq_from_search_data --help
    uv run python -m scripts.verification.apply_tmt_itraq_from_search_data \
        --input-dir <data-root>/lcfm/ \
        --spec PXD001:TMT_6_8_10 \
        --dry-run
    uv run python -m scripts.verification.apply_tmt_itraq_from_search_data \
        --search-data data/search_data.xlsx \
        --input-dir <data-root>/lcfm/ \
        --spec-file tags.yaml
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import partial
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, List, Optional, Set, Tuple, cast

import polars as pl
import typer
import yaml

from scripts.logging_setup import configure_script_logging
from scripts.paths import DEFAULT_SEARCH_DATA
from scripts.preprocessing.parquet_io import atomic_write_parquet, search_data_lookup_key
from scripts.verification.verify_calc_mz import (
    find_parquet_files_in_project,
    is_tmt_quant,
    label_unmodified_lysines,
)

app = typer.Typer(
    help="Apply TMT or iTRAQ lysine labels using search-data Excel",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)

REQUIRED_SEARCH_COLUMNS = [
    "project",
    "file path",
    "acquisition",
    "quant",
    "modifications",
]


class TagKind(str, Enum):
    """Name the isobaric family so command-line and YAML specs map to one UNIMOD accession on lysine."""

    TMT_6_8_10 = "TMT_6_8_10"
    TMT_16_18 = "TMT_16_18"
    ITRAQ = "ITRAQ"


UNIMOD_BY_KIND: Dict[TagKind, str] = {
    TagKind.TMT_6_8_10: "737",
    TagKind.TMT_16_18: "2016",
    TagKind.ITRAQ: "214",
}

# Modifications substrings (case-insensitive). TMT 16/18-plex vs 6/8/10-plex are chosen explicitly.
_TMT_16_18_SUBSTRINGS = ("tmt16", "tmt18")
_TMT_6_8_10_SUBSTRINGS = ("tmt6", "tmt8", "tmt10")

# iTRAQ 4-plex only (case-insensitive). Requires a literal "4" after "itraq" so
# generic "iTRAQ" or 8-plex strings are not matched.
_ITRAQ_4_PLEX_MOD_RE = re.compile(
    r"itraq\s*4(?:\s*-?\s*plex)?",
    re.IGNORECASE,
)


def modifications_match_itraq_4plex(modifications: object) -> bool:
    """Restrict iTRAQ labelling to 4-plex text so 8-plex and generic iTRAQ strings are not tagged.

    Args:
        modifications: Search-data modifications cell.

    Returns:
        True when the text matches iTRAQ 4-plex (``itraq`` + ``4`` + optional plex).
    """
    if modifications is None:
        return False
    return bool(_ITRAQ_4_PLEX_MOD_RE.search(str(modifications)))


def is_itraq_quant(quant: object) -> bool:
    """Treat only an exact ``iTRAQ`` quant cell as iTRAQ (case-insensitive, stripped).

    Args:
        quant: Search-data quant cell.

    Returns:
        True when the value is exactly ``itraq`` after strip and casefold.
    """
    if quant is None:
        return False
    return str(quant).strip().casefold() == "itraq"


def is_dia_acquisition(acquisition: object) -> bool:
    """Split data-independent acquisition (DIA) files from the rest when summarising quant fallback risk.

    Args:
        acquisition: Search-data acquisition cell.

    Returns:
        True when the value is exactly ``DIA`` after strip.
    """
    return acquisition is not None and str(acquisition).strip() == "DIA"


def modifications_match_tag_kind(modifications: object, kind: TagKind) -> bool:
    """Match multiplex-specific modification text so TMT 6/8/10-plex is not confused with 16/18-plex.

    Args:
        modifications: Search-data modifications cell.
        kind: Requested tag family.

    Returns:
        True when modifications indicate that multiplex set (iTRAQ uses 4-plex only).
    """
    if kind == TagKind.ITRAQ:
        return modifications_match_itraq_4plex(modifications)
    s = str(modifications).casefold() if modifications is not None else ""
    if kind == TagKind.TMT_16_18:
        return any(sub in s for sub in _TMT_16_18_SUBSTRINGS)
    if kind == TagKind.TMT_6_8_10:
        return any(sub in s for sub in _TMT_6_8_10_SUBSTRINGS)
    return False


def quant_matches_tag_kind(quant: object, kind: TagKind) -> bool:
    """Use the quant column to distinguish TMT vs iTRAQ families, not the multiplex set.

    Args:
        quant: Search-data quant cell.
        kind: Requested tag family.

    Returns:
        True when quant matches TMT (for either TMT multiplex set) or exact iTRAQ.
    """
    if kind in (TagKind.TMT_6_8_10, TagKind.TMT_16_18):
        return bool(is_tmt_quant(quant))
    return is_itraq_quant(quant)


def row_qualifies_primary(row: dict, kind: TagKind) -> bool:
    """Select files by the strict TMT (modifications plus quant) or iTRAQ (4-plex modifications only) rules.

    Args:
        row: Search-data row as a mapping.
        kind: Requested tag family.

    Returns:
        True when this row is eligible without the quant-only fallback.
    """
    if kind == TagKind.ITRAQ:
        return modifications_match_itraq_4plex(row.get("modifications"))
    return quant_matches_tag_kind(
        row.get("quant"), kind
    ) and modifications_match_tag_kind(row.get("modifications"), kind)


def row_qualifies_fallback_quant(row: dict, kind: TagKind) -> bool:
    """Widen file selection to quant-only when primary multiplex markers are missing.

    Args:
        row: Search-data row as a mapping.
        kind: Requested tag family.

    Returns:
        True when quant matches the tag family.
    """
    return quant_matches_tag_kind(row.get("quant"), kind)


def load_search_data(path: str | Path) -> pl.DataFrame:
    """Load search Excel so file stems can be joined to parquet trees.

    Args:
        path: Search-data workbook with project, raw-filename file path,
            acquisition, quant, and modifications.

    Returns:
        The validated search table.

    Raises:
        ValueError: When required columns are missing.
    """
    df = pl.read_excel(path)
    missing = [c for c in REQUIRED_SEARCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Search data missing columns {missing}. Found: {df.columns}")
    return df


def iter_project_rows(df: pl.DataFrame, project: str) -> Iterable[dict]:
    """Yield only the search rows for one project id (stripped match).

    Args:
        df: Full search-data table.
        project: Project identifier to match.

    Returns:
        Named-row dicts for that project.
    """
    p = project.strip()
    for row in df.iter_rows(named=True):
        if str(row["project"]).strip() != p:
            continue
        yield row


def collect_stems_for_predicate(
    df: pl.DataFrame,
    project: str,
    predicate: Callable[[dict], bool],
) -> Set[str]:
    """Collect unique experiment stems for parquets that pass a TMT/iTRAQ predicate.

    Args:
        df: Search-data table.
        project: Project whose rows are scanned.
        predicate: Row filter (primary or fallback).

    Returns:
        File stems that should receive lysine labelling.
    """
    stems: Set[str] = set()
    for row in iter_project_rows(df, project):
        fp = row.get("file path")
        if fp is None:
            continue
        if not predicate(row):
            continue
        stems.add(search_data_lookup_key(str(fp)))
    return stems


def parquet_paths_for_stems(
    input_dir: Path, project: str, stems: Set[str]
) -> List[Path]:
    """Map search-data stems onto parquet paths that actually exist on disk.

    Args:
        input_dir: Root with per-project parquet subfolders.
        project: Project folder name.
        stems: Experiment stems selected from search data.

    Returns:
        Sorted parquet paths whose stem is in ``stems``.
    """
    if not stems:
        return []
    want = stems
    out: List[Path] = []
    for fp_str in find_parquet_files_in_project(str(input_dir), project):
        p = Path(fp_str)
        stem = search_data_lookup_key(str(p))
        if stem in want:
            out.append(p)
    return sorted(out)


def _normalize_quant_display(quant: object) -> str:
    """Show null/empty quant values as readable tokens in fallback summaries."""
    if quant is None:
        return "(null)"
    return str(quant).strip() or "(empty)"


@dataclass
class QuantSummary:
    """Explain why primary TMT/iTRAQ matching failed so the operator can confirm fallback."""

    quant_row_counts: Dict[str, int]
    rows_dia: int
    rows_non_dia: int
    unique_files_tmt: int
    unique_files_itraq: int
    unique_files_tmt_dia: int
    unique_files_tmt_non_dia: int
    unique_files_itraq_dia: int
    unique_files_itraq_non_dia: int


def _record_quant_file_sets(
    row: dict,
    stem: str,
    dia: bool,
    files_tmt: Set[str],
    files_tmt_dia: Set[str],
    files_tmt_non_dia: Set[str],
    files_itraq: Set[str],
    files_itraq_dia: Set[str],
    files_itraq_non_dia: Set[str],
) -> None:
    """Count unique files by TMT, iTRAQ, and data-independent acquisition so fallback risk is visible."""
    if bool(is_tmt_quant(row.get("quant"))):
        files_tmt.add(stem)
        if dia:
            files_tmt_dia.add(stem)
        else:
            files_tmt_non_dia.add(stem)
    if is_itraq_quant(row.get("quant")):
        files_itraq.add(stem)
        if dia:
            files_itraq_dia.add(stem)
        else:
            files_itraq_non_dia.add(stem)


def build_quant_summary(df: pl.DataFrame, project: str) -> QuantSummary:
    """Build the quantification and acquisition-mode breakdown shown before a project-wide fallback.

    Args:
        df: Search-data table.
        project: Project to summarise.

    Returns:
        Counts that explain whether quant-only labelling is plausible.
    """
    quant_row_counts: Dict[str, int] = {}
    rows_dia = 0
    rows_non_dia = 0
    files_tmt: Set[str] = set()
    files_itraq: Set[str] = set()
    files_tmt_dia: Set[str] = set()
    files_tmt_non_dia: Set[str] = set()
    files_itraq_dia: Set[str] = set()
    files_itraq_non_dia: Set[str] = set()

    for row in iter_project_rows(df, project):
        qdisp = _normalize_quant_display(row.get("quant"))
        quant_row_counts[qdisp] = quant_row_counts.get(qdisp, 0) + 1
        dia = is_dia_acquisition(row.get("acquisition"))
        rows_dia += int(dia)
        rows_non_dia += int(not dia)
        fp = row.get("file path")
        if fp is None:
            continue
        stem = search_data_lookup_key(str(fp))
        _record_quant_file_sets(
            row,
            stem,
            dia,
            files_tmt,
            files_tmt_dia,
            files_tmt_non_dia,
            files_itraq,
            files_itraq_dia,
            files_itraq_non_dia,
        )

    return QuantSummary(
        quant_row_counts=dict(sorted(quant_row_counts.items())),
        rows_dia=rows_dia,
        rows_non_dia=rows_non_dia,
        unique_files_tmt=len(files_tmt),
        unique_files_itraq=len(files_itraq),
        unique_files_tmt_dia=len(files_tmt_dia),
        unique_files_tmt_non_dia=len(files_tmt_non_dia),
        unique_files_itraq_dia=len(files_itraq_dia),
        unique_files_itraq_non_dia=len(files_itraq_non_dia),
    )


def log_quant_summary(project: str, summary: QuantSummary) -> None:
    """Print the fallback quantification summary so a user can confirm project-wide labelling.

    Args:
        project: Project being considered for fallback.
        summary: Counts from ``build_quant_summary``.
    """
    logger.info("Project %s — quant column (row counts by value):", project)
    for qv, n in summary.quant_row_counts.items():
        logger.info("  %s: %d rows", qv, n)
    logger.info(
        "  Rows DIA=%d, non-DIA=%d",
        summary.rows_dia,
        summary.rows_non_dia,
    )
    logger.info(
        "  Unique files with quant=TMT: %d "
        "(with at least one DIA row: %d, at least one non-DIA row: %d)",
        summary.unique_files_tmt,
        summary.unique_files_tmt_dia,
        summary.unique_files_tmt_non_dia,
    )
    logger.info(
        "  Unique files with quant=iTRAQ: %d "
        "(with at least one DIA row: %d, at least one non-DIA row: %d)",
        summary.unique_files_itraq,
        summary.unique_files_itraq_dia,
        summary.unique_files_itraq_non_dia,
    )


def _label_sequence(seq: Optional[str], unimod_id: str) -> Optional[str]:
    """Leave null sequences untouched while tagging unmodified lysines on valid peptides."""
    if seq is None:
        return None
    return label_unmodified_lysines(seq, unimod_id)


def process_parquet_file(
    file_path: Path,
    unimod_id: str,
    dry_run: bool,
) -> Tuple[int, int]:
    """Tag unmodified lysines in one parquet with the UNIMOD accession chosen for this spec.

    Args:
        file_path: Parquet whose ``sequence`` column may contain unmodified lysine.
        unimod_id: UNIMOD accession to write on unmodified lysine (737, 2016, or 214).
        dry_run: Log intended changes without writing.

    Returns:
        ``(1, n)`` if the file was processed (n rows changed), or ``(0, 0)`` if skipped.
    """
    df = pl.read_parquet(file_path)
    if "sequence" not in df.columns:
        logger.warning("Skipping %s: no sequence column", file_path)
        return (0, 0)
    original = df["sequence"]
    new_seq = original.map_elements(
        lambda s: _label_sequence(s, unimod_id),
        return_dtype=pl.String,
    )
    changed = original.is_not_null() & (~original.eq_missing(new_seq))
    n_changed = int(changed.sum())
    if n_changed == 0:
        return (1, 0)
    if dry_run:
        logger.info("[DRY-RUN] Would update %s (%d rows)", file_path, n_changed)
        return (1, n_changed)
    out = df.with_columns(new_seq.alias("sequence"))
    atomic_write_parquet(out, file_path)
    logger.info("Updated %s (%d rows)", file_path, n_changed)
    return (1, n_changed)


def _relpath_or_abs(base: Path, path: Path) -> Path:
    """Log a short path when the parquet sits under the input root."""
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def parse_tag_kind(s: str) -> TagKind:
    """Accept command-line and YAML tag aliases so operators can write TMT_6_8_10 or tmt-6-8-10.

    Args:
        s: Tag kind string from ``--spec`` or YAML.

    Returns:
        The matching ``TagKind``.

    Raises:
        ValueError: When the string is not a known tag kind.
    """
    key = s.strip().upper().replace("-", "_")
    for member in TagKind:
        if member.value == key or member.name == key:
            return member
    raise ValueError(
        f"Unknown tag kind {s!r}. Use one of: " + ", ".join(m.value for m in TagKind)
    )


def parse_spec(spec: str) -> Tuple[str, TagKind]:
    """Parse ``PROJECT:TAG_KIND`` from ``--spec`` so mixed multiplex sets can be listed on the command line.

    Args:
        spec: ``PROJECT:TAG_KIND`` string.

    Returns:
        Project id and tag kind.

    Raises:
        ValueError: When the spec is missing a colon, project, or valid tag.
    """
    if ":" not in spec:
        raise ValueError(f"Invalid --spec {spec!r}; expected PROJECT:TAG_KIND")
    proj, tag = spec.split(":", 1)
    proj = proj.strip()
    if not proj:
        raise ValueError(f"Invalid --spec {spec!r}; empty project")
    return proj, parse_tag_kind(tag.strip())


def _warn_non_itraq_quant_for_itraq_rows(df: pl.DataFrame, project: str) -> None:
    """Warn when 4-plex modifications drive labelling but quantification is not exactly iTRAQ."""
    unexpected: Set[str] = set()
    for row in iter_project_rows(df, project):
        if not modifications_match_itraq_4plex(row.get("modifications")):
            continue
        if is_itraq_quant(row.get("quant")):
            continue
        unexpected.add(_normalize_quant_display(row.get("quant")))
    if unexpected:
        logger.warning(
            "Project %s (ITRAQ): applying K[UNIMOD:214] from iTRAQ 4-plex modifications; "
            "quant is not 'iTRAQ' on some matching rows (seen: %s).",
            project,
            ", ".join(sorted(unexpected)),
        )


def load_specs_from_yaml(path: str | Path) -> List[Tuple[str, TagKind]]:
    """Load a one-tag-per-project YAML map (duplicate keys are invalid YAML; last wins).

    Args:
        path: YAML file of ``project_id: TAG_KIND``.

    Returns:
        Specs in file order.

    Raises:
        ValueError: When the file is not a mapping or a tag is unknown.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML spec file must be a mapping project -> tag: {path}")
    out: List[Tuple[str, TagKind]] = []
    for k, v in data.items():
        proj = str(k).strip()
        out.append((proj, parse_tag_kind(str(v).strip())))
    return out


def _primary_or_fallback_stems(
    df: pl.DataFrame,
    project: str,
    kind: TagKind,
    unimod: str,
    confirm_project_wide: Set[str],
) -> Tuple[Set[str], str, int]:
    """Prefer multiplex-aware file selection; only fall back to quant-only after confirmation."""
    pred_primary = cast(
        Callable[[dict], bool],
        partial(row_qualifies_primary, kind=kind),
    )
    primary = collect_stems_for_predicate(df, project, pred_primary)
    if primary:
        if kind == TagKind.ITRAQ:
            _warn_non_itraq_quant_for_itraq_rows(df, project)
            return primary, "primary (iTRAQ 4-plex modifications)", 0
        return primary, "primary (modifications plus quant)", 0

    logger.warning(
        "Project %s (%s): no search rows matched primary rules for this tag.",
        project,
        kind.value,
    )
    log_quant_summary(project, build_quant_summary(df, project))

    pred_fb = cast(
        Callable[[dict], bool],
        partial(row_qualifies_fallback_quant, kind=kind),
    )
    fallback = collect_stems_for_predicate(df, project, pred_fb)
    if not fallback:
        logger.error(
            "No files with matching quant type for %s in project %s; skip.",
            kind.value,
            project,
        )
        return set(), "", 1

    if project not in confirm_project_wide:
        msg = (
            f"Apply {kind.value} (UNIMOD {unimod}) project-wide by quant only "
            f"to {len(fallback)} file stem(s) under {project}? "
            f"(Parquets for stems with quant matching this tag, including DIA.)"
        )
        if not typer.confirm(msg, default=False):
            logger.error(
                "Aborted %s — use --confirm-project-wide %s",
                project,
                project,
            )
            return set(), "", 1

    return fallback, "fallback (quant-only, confirmed)", 0


def run_apply_labels(
    search_data: Path,
    input_dir: Path,
    specs: List[Tuple[str, TagKind]],
    *,
    dry_run: bool,
    confirm_project_wide: Set[str],
) -> int:
    """Apply each project/tag spec, aborting unconfirmed quant-only fallbacks.

    Args:
        search_data: Search Excel used to pick file stems.
        input_dir: Root with per-project parquet subfolders.
        specs: ``(project, tag kind)`` pairs from the command line and/or YAML.
        dry_run: Preview writes without modifying files.
        confirm_project_wide: Project ids allowed to skip the fallback prompt.

    Returns:
        0 on full success, 1 if any spec was skipped or had no matching parquets.
    """
    df = load_search_data(search_data)
    input_path = Path(input_dir)
    exit_code = 0
    unimod_ids = {k: UNIMOD_BY_KIND[k] for k in TagKind}

    for project, kind in specs:
        unimod = unimod_ids[kind]
        stems, mode, err = _primary_or_fallback_stems(
            df, project, kind, unimod, confirm_project_wide
        )
        if err:
            exit_code = 1
            continue

        paths = parquet_paths_for_stems(input_path, project, stems)
        if not paths:
            logger.warning(
                "Project %s: no parquet files on disk for %d stem(s); check input-dir",
                project,
                len(stems),
            )
            exit_code = 1
            continue

        logger.info(
            "Project %s — %s — %d parquet(s), UNIMOD %s",
            project,
            mode,
            len(paths),
            unimod,
        )
        for pth in paths:
            logger.info("  %s", _relpath_or_abs(input_path, pth))

        total_rows = 0
        for pth in paths:
            _, n = process_parquet_file(pth, unimod, dry_run=dry_run)
            total_rows += n
        logger.info(
            "Project %s done: parquets=%d rows_with_K_changes=%d dry_run=%s",
            project,
            len(paths),
            total_rows,
            dry_run,
        )

    return exit_code


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Root directory with per-project parquet subfolders",
        ),
    ],
    search_data: Annotated[
        Path,
        typer.Option(
            "--search-data",
            "-s",
            help=(
                "Search-data Excel with project, raw-filename file path, "
                "acquisition, quant, and modifications"
            ),
        ),
    ] = DEFAULT_SEARCH_DATA,
    spec: Annotated[
        Optional[List[str]],
        typer.Option(
            "--spec",
            help="PROJECT:TAG_KIND (repeat). TAG_KIND: TMT_6_8_10, TMT_16_18, ITRAQ",
        ),
    ] = None,
    spec_file: Annotated[
        Optional[Path],
        typer.Option(
            "--spec-file",
            help=(
                "YAML mapping project id -> tag kind (one entry per project; "
                "duplicate keys invalid — use repeated --spec for mixed tags)"
            ),
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n")] = False,
    confirm_project_wide: Annotated[
        Optional[List[str]],
        typer.Option(
            "--confirm-project-wide",
            help="Project id (repeat) to allow quant-only fallback without prompt",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging"),
    ] = False,
) -> None:
    """Label unmodified lysines with TMT or iTRAQ UNIMOD accessions using search-data multiplex rules (and optional quant fallback).

    Args:
        input_dir: Root directory with per-project parquet subfolders.
        search_data: Search-data Excel with project, raw-filename file path,
            acquisition, quant, and modifications.
        spec: Repeatable ``PROJECT:TAG_KIND`` (TMT_6_8_10, TMT_16_18, ITRAQ).
        spec_file: YAML mapping project id to tag kind (one entry per project).
        dry_run: Log actions without writing files.
        confirm_project_wide: Project ids that may use quant-only fallback without a prompt.
        verbose: Enable DEBUG logging.
    """
    configure_script_logging(verbose=verbose)
    spec_list = spec or []
    confirm_list = confirm_project_wide or []

    specs: List[Tuple[str, TagKind]] = []
    if spec_file is not None:
        specs.extend(load_specs_from_yaml(spec_file))
    for s in spec_list:
        specs.append(parse_spec(s))
    if not specs:
        raise typer.BadParameter("Provide --spec and/or --spec-file")

    confirm_set = {p.strip() for p in confirm_list if p.strip()}
    code = run_apply_labels(
        search_data,
        input_dir,
        specs,
        dry_run=dry_run,
        confirm_project_wide=confirm_set,
    )
    if code != 0:
        raise typer.Exit(code)


if __name__ == "__main__":
    app()
