"""Build combined cross-set parquet from paired query/library files (Kostas protocol).

Supports IPC (``.ipc``, ``.mzML.ipc``) and parquet inputs. For each basename-matched
ACFM/LCFM file pair: queries = ``ACFM/file − LCFM/file`` (diff by scan/usi/source_file),
embed the full LCFM file + ACFM diff, and mark spectra of the top-N peptides selected
from LCFM-valid (or the LCFM file itself when no valid split exists for the project)
as retrieval anchors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import polars as pl

IPC_COLUMN_REMAP: dict[str, str] = {
    "mz": "mz_array",
    "intensity": "intensity_array",
    "rt": "retention_time",
}

# When several peptide columns exist, prefer modified peptidoforms for ``sequence``.
SEQUENCE_SOURCE_PRIORITY: tuple[str, ...] = (
    "modified_sequence",
    "Modified sequence",
    "modified_peptide",
    "sequence",
    "Sequence",
    "peptide",
)

REQUIRED_SPECTRUM_COLUMNS = ("mz_array", "intensity_array")
OPTIONAL_NUMERIC_COLUMNS = (
    "precursor_mz",
    "precursor_charge",
    "scan",
    "retention_time",
    "collision_energy",
    "frag_type",
    "source_file",
    "header",
    "index",
    "hyperscore",
    "probability",
    "expectation",
    "pair_file",
    "is_selected_anchor",
)
UNMODIFIED_PEPTIDE_SOURCE_PRIORITY: tuple[str, ...] = (
    "unmodified_peptide",
    "peptide",
    "Sequence",
)

ANNOTATION_COLUMNS = ("sequence", "unmodified_peptide", "usi")
SCORE_COLUMNS = ("hyperscore", "probability", "expectation")


def discover_data_files(directory: str | Path) -> list[Path]:
    """Return sorted IPC/parquet files under *directory* (non-recursive)."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("*.ipc", "*.mzML.ipc", "*.parquet"):
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No .ipc or .parquet files found in {root}")
    return paths


def _scan_file(path: Path) -> pl.LazyFrame:
    if path.suffix == ".parquet" or path.name.endswith(".parquet"):
        lf = pl.scan_parquet(str(path))
    else:
        lf = pl.scan_ipc(str(path))
    return lf.with_columns(pl.lit(path.name).alias("_input_file"))


def _read_file_schema(path: Path) -> dict[str, pl.DataType]:
    """Read parquet/IPC schema without loading the full table."""
    if str(path).endswith(".parquet"):
        try:
            return dict(pl.read_schema(str(path)))
        except AttributeError:
            return dict(pl.scan_parquet(str(path)).collect_schema())
    return dict(pl.read_ipc_schema(str(path)))


def _blank_to_null(expr: pl.Expr) -> pl.Expr:
    as_str = expr.cast(pl.Utf8)
    return (
        pl.when(as_str.is_null() | as_str.is_in(["", "None", "nan", "NaN"]))
        .then(None)
        .otherwise(as_str)
    )


def _sequence_column_expr(schema: dict[str, pl.DataType]) -> pl.Expr | None:
    """Coalesce peptide columns so null ``modified_peptide`` falls back to plain peptide."""
    available = [source for source in SEQUENCE_SOURCE_PRIORITY if source in schema]
    if not available:
        return None
    return pl.coalesce([_blank_to_null(pl.col(source)) for source in available]).alias("sequence")


def _rename_available_columns(lf: pl.LazyFrame, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Rename IPC columns to canonical names (at most one source per target)."""
    exprs: list[pl.Expr] = []
    for source, target in IPC_COLUMN_REMAP.items():
        if source in schema and target not in schema:
            exprs.append(pl.col(source).alias(target))
    sequence_expr = _sequence_column_expr(schema)
    if sequence_expr is not None:
        exprs.append(sequence_expr)
    if exprs:
        lf = lf.with_columns(exprs)
    return lf


def _ensure_source_file(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    if "source_file" not in schema:
        return lf.with_columns(pl.col("_input_file").alias("source_file"))
    return lf.with_columns(pl.coalesce([pl.col("source_file"), pl.col("_input_file")]).alias("source_file"))


def _unmodified_peptide_column_expr(schema: dict[str, pl.DataType]) -> pl.Expr | None:
    """Return the highest-priority unmodified peptide column."""
    for source in UNMODIFIED_PEPTIDE_SOURCE_PRIORITY:
        if source in schema:
            return pl.col(source).alias("unmodified_peptide")
    return None


def _ensure_sequence_columns(lf: pl.LazyFrame, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    if "sequence" not in schema and _sequence_column_expr(schema) is None:
        lf = lf.with_columns(pl.lit("").alias("sequence"))
    if "unmodified_peptide" not in schema:
        unmodified_expr = _unmodified_peptide_column_expr(schema)
        if unmodified_expr is not None:
            lf = lf.with_columns(unmodified_expr)
        else:
            lf = lf.with_columns(pl.lit("").alias("unmodified_peptide"))
    # Prefer a non-empty sequence; fall back to unmodified peptide (common when
    # modified_peptide is null for many rows).
    names = set(lf.collect_schema().names())
    if "sequence" in names and "unmodified_peptide" in names:
        lf = lf.with_columns(
            pl.coalesce(
                [
                    _blank_to_null(pl.col("sequence")),
                    _blank_to_null(pl.col("unmodified_peptide")),
                    pl.lit(""),
                ]
            ).alias("sequence")
        )
    return lf


def _ensure_usi(lf: pl.LazyFrame, schema: dict[str, pl.DataType], project_id: str | None) -> pl.LazyFrame:
    if "usi" in schema:
        return lf
    project = project_id or "crossset"
    if "scan" in schema:
        return lf.with_columns(
            pl.format(
                "mzspec:{}:{}:scan:{}",
                pl.lit(project),
                pl.col("source_file"),
                pl.col("scan").cast(pl.Utf8),
            ).alias("usi")
        )
    return lf.with_columns(pl.format("mzspec:{}:{}:index:{}", pl.lit(project), pl.col("source_file"), pl.col("index").cast(pl.Utf8)).alias("usi"))


def _overlap_id_expr(overlap_key: str) -> pl.Expr:
    if overlap_key == "usi":
        return pl.col("usi").cast(pl.Utf8)
    if overlap_key == "scan":
        return pl.concat_str([pl.col("source_file").fill_null(""), pl.lit(":"), pl.col("scan").cast(pl.Utf8)])
    if overlap_key == "source_file":
        return pl.col("source_file").cast(pl.Utf8)
    return pl.col(overlap_key).cast(pl.Utf8)


def _prepare_lazy_frame(path: Path, *, project_id: str | None) -> pl.LazyFrame:
    schema = _read_file_schema(path)
    lf = _scan_file(path)
    lf = _rename_available_columns(lf, schema)
    lf = _ensure_source_file(lf)
    current_schema = lf.collect_schema()
    lf = _ensure_sequence_columns(lf, current_schema)
    lf = _ensure_usi(lf, lf.collect_schema(), project_id)
    return lf


def _select_common_columns(lfs: Iterable[pl.LazyFrame]) -> list[str]:
    columns: set[str] | None = None
    for lf in lfs:
        names = set(lf.collect_schema().names())
        columns = names if columns is None else columns & names
    if not columns:
        raise ValueError("No common columns between query and library inputs")
    keep = list(REQUIRED_SPECTRUM_COLUMNS) + [c for c in OPTIONAL_NUMERIC_COLUMNS if c in columns]
    keep += [c for c in ANNOTATION_COLUMNS if c in columns]
    keep += ["search_tier", "overlap_id", "_input_file"]
    return list(dict.fromkeys(keep))


def discover_paired_files(
    acfm_dir: str | Path,
    lcfm_dir: str | Path,
    *,
    file_name: str | None = None,
) -> list[tuple[Path, Path]]:
    """Return ``(acfm_path, lcfm_path)`` pairs matched by basename."""
    acfm_paths = {path.name: path for path in discover_data_files(acfm_dir)}
    lcfm_paths = {path.name: path for path in discover_data_files(lcfm_dir)}
    names = sorted(set(acfm_paths) & set(lcfm_paths))
    if file_name is not None:
        if file_name not in acfm_paths or file_name not in lcfm_paths:
            raise FileNotFoundError(
                f"Paired file {file_name!r} not found in both ACFM and LCFM dirs. "
                f"ACFM has it: {file_name in acfm_paths}; LCFM has it: {file_name in lcfm_paths}"
            )
        names = [file_name]
    if not names:
        raise FileNotFoundError(f"No basename-matched files between {acfm_dir} and {lcfm_dir}")
    return [(acfm_paths[name], lcfm_paths[name]) for name in names]


def select_top_peptides(
    lf: pl.LazyFrame,
    *,
    top_n: int | None = 10,
    peptide_col: str = "sequence",
) -> list[str]:
    """Rank peptides by support, then engine score / confidence (Kostas criteria).

    ``top_n=None`` returns every distinct peptide (still ranked, no cap) — use this
    for an "all peptides" anchor run instead of a fixed top-N shortlist.
    """
    schema = lf.collect_schema()
    if peptide_col not in schema.names():
        raise ValueError(f"Cannot rank peptides: missing column {peptide_col!r}")

    aggs: list[pl.Expr] = [pl.len().alias("n_support")]
    sort_cols = ["n_support"]
    if "hyperscore" in schema.names():
        aggs.append(pl.col("hyperscore").max().alias("max_hyperscore"))
        sort_cols.append("max_hyperscore")
    if "probability" in schema.names():
        aggs.append(pl.col("probability").max().alias("max_probability"))
        sort_cols.append("max_probability")
    if "expectation" in schema.names():
        aggs.append((-pl.col("expectation").min()).alias("neg_min_expectation"))
        sort_cols.append("neg_min_expectation")

    ranked_lf = (
        lf.select(
            [
                _blank_to_null(pl.col(peptide_col)).alias("peptide_id"),
                *[pl.col(c) for c in SCORE_COLUMNS if c in schema.names()],
            ]
        )
        .filter(pl.col("peptide_id").is_not_null())
        .group_by("peptide_id")
        .agg(aggs)
        .sort(sort_cols, descending=True)
    )
    if top_n is not None:
        ranked_lf = ranked_lf.head(top_n)
    ranked = ranked_lf.collect()
    return ranked["peptide_id"].to_list()


def load_valid_experiment_stems(
    valid_glob: str,
    *,
    project_id: str,
) -> set[str]:
    """Return experiment stems present in LCFM-valid for *project_id*."""
    lf = pl.scan_parquet(valid_glob).filter(pl.col("usi").str.contains(project_id))
    schema = lf.collect_schema()
    if "experiment_name" not in schema.names():
        return set()
    return set(lf.select(pl.col("experiment_name").unique()).collect()["experiment_name"].to_list())


def load_valid_peptide_rows(
    valid_glob: str,
    *,
    project_id: str,
    experiment_stem: str,
) -> pl.LazyFrame | None:
    """LCFM-valid rows for one experiment/file stem, or None if none match."""
    lf = (
        pl.scan_parquet(valid_glob)
        .filter(pl.col("usi").str.contains(project_id))
        .filter(pl.col("experiment_name") == experiment_stem)
    )
    n = lf.select(pl.len()).collect().item()
    return lf if n > 0 else None


def build_kostas_file_pair(
    *,
    acfm_path: Path,
    lcfm_path: Path,
    overlap_key: str = "scan",
    project_id: str | None = None,
    top_n_peptides: int | None = 10,
    valid_glob: str | None = None,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Build one Kostas file-pair lazy frame (ACFM diff queries + full LCFM file).

    Protocol:
    - queries = ACFM/file − LCFM/file (overlap on *overlap_key*)
    - embed all LCFM/file spectra + ACFM diff
    - select top-N peptides from LCFM-valid for that file when *valid_glob* matches;
      otherwise rank on all labeled spectra in the LCFM file
    - mark LCFM spectra of those peptides with ``is_selected_anchor=1``

    ``top_n_peptides=None`` selects *every* distinct peptide as an anchor (no cap),
    i.e. retrieval runs against the full LCFM library for that file pair.
    """
    project = project_id or infer_project_id_from_path(acfm_path) or infer_project_id_from_path(lcfm_path)
    pair_name = acfm_path.name
    experiment_stem = Path(pair_name).stem.replace(".mzML", "")

    acfm_lf = _prepare_lazy_frame(acfm_path, project_id=project)
    lcfm_lf = _prepare_lazy_frame(lcfm_path, project_id=project)

    acfm_lf = acfm_lf.with_columns(
        _overlap_id_expr(overlap_key).alias("overlap_id"),
        pl.lit(pair_name).alias("pair_file"),
    )
    lcfm_lf = lcfm_lf.with_columns(
        _overlap_id_expr(overlap_key).alias("overlap_id"),
        pl.lit(pair_name).alias("pair_file"),
    )

    lcfm_keys = (
        lcfm_lf.select(pl.col("overlap_id"))
        .filter(pl.col("overlap_id").is_not_null())
        .unique()
        .collect(engine="streaming")["overlap_id"]
        .to_list()
    )
    acfm_diff = acfm_lf.filter(~pl.col("overlap_id").is_in(lcfm_keys))

    ranking_lf: pl.LazyFrame | None = None
    ranking_source = "lcfm_file"
    if valid_glob:
        ranking_lf = load_valid_peptide_rows(
            valid_glob,
            project_id=project or "",
            experiment_stem=experiment_stem,
        )
        if ranking_lf is not None:
            ranking_source = "lcfm_valid"

    if ranking_lf is None:
        ranking_lf = lcfm_lf

    top_peptides = select_top_peptides(ranking_lf, top_n=top_n_peptides, peptide_col="sequence")
    if not top_peptides:
        # Fall back to unmodified peptide ids if sequence ranking is empty.
        top_peptides = select_top_peptides(
            ranking_lf,
            top_n=top_n_peptides,
            peptide_col="unmodified_peptide",
        )

    lcfm_out = lcfm_lf.with_columns(
        pl.lit("lcfm").alias("search_tier"),
        pl.when(pl.col("sequence").is_in(top_peptides))
        .then(pl.lit("1"))
        .otherwise(pl.lit("0"))
        .alias("is_selected_anchor"),
    )
    acfm_out = acfm_diff.with_columns(
        pl.lit("acfm").alias("search_tier"),
        pl.lit("0").alias("is_selected_anchor"),
    )

    common_cols = _select_common_columns([acfm_out, lcfm_out])
    acfm_out = acfm_out.select([c for c in common_cols if c in acfm_out.collect_schema().names()])
    lcfm_out = lcfm_out.select([c for c in common_cols if c in lcfm_out.collect_schema().names()])
    combined = pl.concat([acfm_out, lcfm_out], how="vertical_relaxed")

    meta = {
        "pair_file": pair_name,
        "num_acfm_raw": int(acfm_lf.select(pl.len()).collect().item()),
        "num_lcfm": int(lcfm_lf.select(pl.len()).collect().item()),
        "num_acfm_diff": int(acfm_diff.select(pl.len()).collect().item()),
        "num_exclude_keys": len(lcfm_keys),
        "top_peptides": top_peptides,
        "num_selected_anchor_spectra": int(
            lcfm_out.filter(pl.col("is_selected_anchor") == "1").select(pl.len()).collect().item()
        ),
        "peptide_ranking_source": ranking_source,
        "overlap_key": overlap_key,
    }
    return combined, meta


def build_kostas_protocol_parquet(
    *,
    acfm_dir: str | Path,
    lcfm_dir: str | Path,
    output_path: str | Path,
    overlap_key: str = "scan",
    project_id: str | None = None,
    top_n_peptides: int | None = 10,
    valid_glob: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    """Run Kostas protocol over one file or all basename-paired files.

    ``top_n_peptides=None`` selects every distinct peptide per file pair as an
    anchor (no cap) instead of a fixed top-N shortlist.
    """
    project = project_id or infer_project_id_from_path(acfm_dir) or infer_project_id_from_path(lcfm_dir)
    pairs = discover_paired_files(acfm_dir, lcfm_dir, file_name=file_name)

    frames: list[pl.LazyFrame] = []
    pair_summaries: list[dict[str, Any]] = []
    for acfm_path, lcfm_path in pairs:
        frame, meta = build_kostas_file_pair(
            acfm_path=acfm_path,
            lcfm_path=lcfm_path,
            overlap_key=overlap_key,
            project_id=project,
            top_n_peptides=top_n_peptides,
            valid_glob=valid_glob,
        )
        frames.append(frame)
        pair_summaries.append(meta)

    combined = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.sink_parquet(str(output))

    summary = {
        "protocol": "kostas",
        "output_path": str(output),
        "project_id": project,
        "overlap_key": overlap_key,
        "top_n_peptides": top_n_peptides,
        "valid_glob": valid_glob,
        "num_pairs": len(pairs),
        "num_queries": int(
            combined.filter(pl.col("search_tier") == "acfm").select(pl.len()).collect().item()
        ),
        "num_library_all_lcfm": int(
            combined.filter(pl.col("search_tier") == "lcfm").select(pl.len()).collect().item()
        ),
        "num_selected_anchor_spectra": int(
            combined.filter(pl.col("is_selected_anchor") == "1").select(pl.len()).collect().item()
        ),
        "pairs": pair_summaries,
    }
    summary_path = output.with_suffix(".kostas_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary


def infer_project_id_from_path(path: str | Path) -> str | None:
    """Extract PXD project id from a folder path when present."""
    match = re.search(r"(PXD\d+)", str(path))
    return match.group(1) if match else None
