r"""Apply TMT or iTRAQ lysine labels from search-data Excel.

For each user-specified (project, tag kind), labels bare K in parquet sequences with
the matching UNIMOD (TMT 6/8/10 → 737, TMT 16/18 → 2016, iTRAQ-4plex → 214).

**TMT** (primary): each row must match both ``quant`` = TMT and TMT-plex markers in
``modifications`` (see ``modifications_match_tag_kind``).

**iTRAQ** (primary): only ``modifications`` are used — the row must match **iTRAQ
4-plex** text (``itraq`` + ``4`` + optional ``plex`` / spacing; see
``modifications_match_itraq_4plex``). Other iTRAQ plexes are not matched. If
``quant`` is not exactly ``iTRAQ`` (case-insensitive), a **warning** is logged but
labeling still proceeds for those rows.

If no row qualifies on the primary rules, the script prints a quant summary and may
require ``--confirm-project-wide PROJECT`` or interactive confirmation before a
**quant-only** fallback (TMT or ``quant`` = iTRAQ for ITRAQ specs).

**YAML spec file:** must be a single mapping ``project_id: TAG_KIND``. Duplicate
project keys are invalid in YAML (the last wins). To run the same project with more
than one tag kind (e.g. mixed TMT 10-plex and 16-plex files), use repeated
``--spec PROJECT:TAG`` on the CLI and/or split entries across multiple YAML files.

USAGE:
======
python scripts/verification/apply_tmt_itraq_from_search_data.py \\
    --search-data search_data.xlsx \\
    --input-dir <data-root>/lcfm/ \\
    --spec PXD001:TMT_6_8_10 \\
    --dry-run

python scripts/verification/apply_tmt_itraq_from_search_data.py \\
    --search-data search_data.xlsx \\
    --input-dir <data-root>/lcfm/ \\
    --spec-file tags.yaml
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from functools import partial
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, List, Optional, Set, Tuple, cast

import polars as pl
import typer
import yaml

from scripts.verification.verify_calc_mz import (
    extract_file_name,
    find_parquet_files_in_project,
    is_tmt_quant,
    label_unmodified_lysines,
)

app = typer.Typer(help="Apply TMT/iTRAQ K-labels using search-data Excel")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    """Isobaric label family for CLI specs and UNIMOD mapping."""

    TMT_6_8_10 = "TMT_6_8_10"
    TMT_16_18 = "TMT_16_18"
    ITRAQ = "ITRAQ"


UNIMOD_BY_KIND: Dict[TagKind, str] = {
    TagKind.TMT_6_8_10: "737",
    TagKind.TMT_16_18: "2016",
    TagKind.ITRAQ: "214",
}

# Modifications substrings (case-insensitive). TMT 16/18 vs 6/8/10 are chosen explicitly.
_TMT_16_18_SUBSTRINGS = ("tmt16", "tmt18")
_TMT_6_8_10_SUBSTRINGS = ("tmt6", "tmt8", "tmt10")

# iTRAQ 4-plex only (case-insensitive). Requires a literal "4" after "itraq" so
# generic "iTRAQ" or 8-plex strings are not matched.
_ITRAQ_4_PLEX_MOD_RE = re.compile(
    r"itraq\s*4(?:\s*-?\s*plex)?",
    re.IGNORECASE,
)


def modifications_match_itraq_4plex(modifications: object) -> bool:
    """True if modifications indicate iTRAQ 4-plex labeling (not other iTRAQ plexes)."""
    if modifications is None:
        return False
    return bool(_ITRAQ_4_PLEX_MOD_RE.search(str(modifications)))


def is_itraq_quant(quant: object) -> bool:
    """True if search-data quant value indicates iTRAQ (exact strip, case-insensitive)."""
    if quant is None:
        return False
    return str(quant).strip().casefold() == "itraq"


def is_dia_acquisition(acquisition: object) -> bool:
    """True if acquisition column indicates DIA."""
    return acquisition is not None and str(acquisition).strip() == "DIA"


def modifications_match_tag_kind(modifications: object, kind: TagKind) -> bool:
    """True if modifications string matches the given tag kind (case-insensitive)."""
    if kind == TagKind.ITRAQ:
        return modifications_match_itraq_4plex(modifications)
    s = str(modifications).casefold() if modifications is not None else ""
    if kind == TagKind.TMT_16_18:
        return any(sub in s for sub in _TMT_16_18_SUBSTRINGS)
    if kind == TagKind.TMT_6_8_10:
        return any(sub in s for sub in _TMT_6_8_10_SUBSTRINGS)
    return False


def quant_matches_tag_kind(quant: object, kind: TagKind) -> bool:
    """True if quant column matches the tag family (TMT vs iTRAQ)."""
    if kind in (TagKind.TMT_6_8_10, TagKind.TMT_16_18):
        return bool(is_tmt_quant(quant))
    return is_itraq_quant(quant)


def row_qualifies_primary(row: dict, kind: TagKind) -> bool:
    """True if row passes primary file selection (ITRAQ: mods only; TMT: mods+quant)."""
    if kind == TagKind.ITRAQ:
        return modifications_match_itraq_4plex(row.get("modifications"))
    return quant_matches_tag_kind(
        row.get("quant"), kind
    ) and modifications_match_tag_kind(row.get("modifications"), kind)


def row_qualifies_fallback_quant(row: dict, kind: TagKind) -> bool:
    """True if row quant matches tag family (fallback when mods do not match)."""
    return quant_matches_tag_kind(row.get("quant"), kind)


def load_search_data(path: str | Path) -> pl.DataFrame:
    """Load search Excel and validate required columns."""
    df = pl.read_excel(path)
    missing = [c for c in REQUIRED_SEARCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Search data missing columns {missing}. Found: {df.columns}")
    return df


def iter_project_rows(df: pl.DataFrame, project: str) -> Iterable[dict]:
    """Yield search rows for a single project id (stripped match)."""
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
    """Collect unique file stems for rows in project where predicate(row) is true."""
    stems: Set[str] = set()
    for row in iter_project_rows(df, project):
        fp = row.get("file path")
        if fp is None:
            continue
        if not predicate(row):
            continue
        stems.add(extract_file_name(str(fp)))
    return stems


def parquet_paths_for_stems(
    input_dir: Path, project: str, stems: Set[str]
) -> List[Path]:
    """List parquet paths under input_dir/project whose stem is in stems."""
    if not stems:
        return []
    want = stems
    out: List[Path] = []
    for fp_str in find_parquet_files_in_project(str(input_dir), project):
        p = Path(fp_str)
        stem = extract_file_name(str(p))
        if stem in want:
            out.append(p)
    return sorted(out)


def _normalize_quant_display(quant: object) -> str:
    if quant is None:
        return "(null)"
    return str(quant).strip() or "(empty)"


@dataclass
class QuantSummary:
    """Per-project quant breakdown for user messaging."""

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
    """Aggregate quant values and per-file TMT/iTRAQ counts for one project."""
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
        stem = extract_file_name(str(fp))
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
    """Log quant breakdown from build_quant_summary."""
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


def _atomic_write_parquet(df: pl.DataFrame, file_path: Path) -> None:
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".parquet", dir=file_path.parent)
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _label_sequence(seq: Optional[str], unimod_id: str) -> Optional[str]:
    if seq is None:
        return None
    return label_unmodified_lysines(seq, unimod_id)


def process_parquet_file(
    file_path: Path,
    unimod_id: str,
    dry_run: bool,
) -> Tuple[int, int]:
    """Return (files_processed 0|1, rows_changed)."""
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
    _atomic_write_parquet(out, file_path)
    logger.info("Updated %s (%d rows)", file_path, n_changed)
    return (1, n_changed)


def _relpath_or_abs(base: Path, path: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def parse_tag_kind(s: str) -> TagKind:
    """Parse CLI/YAML tag string into TagKind."""
    key = s.strip().upper().replace("-", "_")
    for member in TagKind:
        if member.value == key or member.name == key:
            return member
    raise ValueError(
        f"Unknown tag kind {s!r}. Use one of: " + ", ".join(m.value for m in TagKind)
    )


def parse_spec(spec: str) -> Tuple[str, TagKind]:
    """Parse ``PROJECT:TAG_KIND`` from ``--spec``."""
    if ":" not in spec:
        raise ValueError(f"Invalid --spec {spec!r}; expected PROJECT:TAG_KIND")
    proj, tag = spec.split(":", 1)
    proj = proj.strip()
    if not proj:
        raise ValueError(f"Invalid --spec {spec!r}; empty project")
    return proj, parse_tag_kind(tag.strip())


def _warn_non_itraq_quant_for_itraq_rows(df: pl.DataFrame, project: str) -> None:
    """Log when iTRAQ-4plex mods are used but quant is not exactly 'iTRAQ'."""
    unexpected: Set[str] = set()
    for row in iter_project_rows(df, project):
        if not modifications_match_itraq_4plex(row.get("modifications")):
            continue
        if is_itraq_quant(row.get("quant")):
            continue
        unexpected.add(_normalize_quant_display(row.get("quant")))
    if unexpected:
        logger.warning(
            "Project %s (ITRAQ): applying K[UNIMOD:214] from iTRAQ-4plex modifications; "
            "quant is not 'iTRAQ' on some matching rows (seen: %s).",
            project,
            ", ".join(sorted(unexpected)),
        )


def load_specs_from_yaml(path: str | Path) -> List[Tuple[str, TagKind]]:
    """Load project → tag kind mapping from YAML (one tag per project key; no duplicate keys)."""
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
    """Return (stems, mode, err). err is 1 if this spec should be skipped."""
    pred_primary = cast(
        Callable[[dict], bool],
        partial(row_qualifies_primary, kind=kind),
    )
    primary = collect_stems_for_predicate(df, project, pred_primary)
    if primary:
        if kind == TagKind.ITRAQ:
            _warn_non_itraq_quant_for_itraq_rows(df, project)
            return primary, "primary (iTRAQ-4plex mods)", 0
        return primary, "primary (mods+quant)", 0

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
            f"(Parquets for stems with quant matching this tag, incl. DIA.)"
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
    """Return 0 on success, 1 if any spec aborted without confirmation."""
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
    search_data: Annotated[
        Path,
        typer.Option(
            "--search-data",
            "-s",
            help="Search data Excel (project, file path, acquisition, quant, modifications)",
        ),
    ],
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Root directory with per-project parquet subfolders",
        ),
    ],
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
) -> None:
    """Label bare K with TMT or iTRAQ UNIMOD from search-data rules."""
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
