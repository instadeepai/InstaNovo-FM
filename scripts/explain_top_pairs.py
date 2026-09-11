# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Explain the most-similar cross-set query->library pairs with MCP match-quality views.

For each of the top-N query->library pairs (ranked by embedding cosine similarity from
Stage 1), this renders the proteomics-mcp dashboards that explain *why* the transfer is
(or isn't) chemically supported -- there is no query sequence / ground truth, so the goal
is explanation of the most-similar cases, not a rescue-rate estimate.

Per pair it produces:
  - query_vs_peptide_mirror.html   : query observed vs the transferred library peptide's
                                     theoretical b/y ions, with the full Match Quality /
                                     Ion Evidence / Mass Error / Candidate metric tiles
                                     (this is "block B" -- does the peptide explain the
                                     query spectrum?).
  - query_annotated.html           : query observed peaks labeled with the peptide's ions.
  - library_vs_peptide_mirror.html : library observed vs the same peptide's theoretical
                                     ions ("block C" self-consistency reference -- the
                                     peptide should explain its *own* library spectrum).
  - library_annotated.html         : library observed peaks labeled with the peptide.
  - index.html                     : summary table (embedding score/margin, scan gap,
                                     block A observed-vs-observed cosine, block B/C fit)
                                     linking every per-pair view.

Theoretical fragments are used for the predicted side (offline, deterministic); MS2PIP is
not required. Block A (query observed vs library observed) is the direct spectrum-to-
spectrum similarity that the embedding is implicitly matching on.

Usage:
    uv run python scripts/explain_top_pairs.py \\
        --candidates-csv stage2_477_1/cross_set_topk_candidates.csv \\
        --combined-parquet stage2_477_1/PXD074343_477-1_all.parquet \\
        --output-dir stage2_477_1/top_pairs --top-n 5 --label "PXD074343 477-1"
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from instanovo_fm.eval.spectrum_metrics.mcp_scoring import (
    MCP_AVAILABLE,
    build_predicted_spectrum,
)
from instanovo_fm.eval.spectrum_metrics.worker import (
    LibrarySelfWorkItem,
    QueryRankWorkItem,
    SpectrumRecord,
    score_library_self,
    score_query_rank,
)

# Fully explicit comparison names for the results table (no "block A/B/C" jargon).
CMP_A = "query_observed_vs_library_observed"                       # direct spectrum-to-spectrum similarity
CMP_B = "query_observed_vs_transferred_peptide_theoretical"        # does the transferred peptide explain the query?
CMP_C = "library_observed_vs_peptide_theoretical"                  # library self-consistency reference

# Rename the worker's internal short prefixes to the explicit comparison names.
_PREFIX_RENAME = {
    "q_obs__lib_obs__": f"{CMP_A}__",
    "q_obs__lib_theo__": f"{CMP_B}__",
    "lib_obs__lib_theo__": f"{CMP_C}__",
}


def _explicit_key(key: str) -> str:
    """Map a worker metric key (e.g. ``q_obs__lib_theo__hyperscore``) to an explicit name."""
    for short, explicit in _PREFIX_RENAME.items():
        if key.startswith(short):
            return explicit + key[len(short):]
    return key


# Intensity-similarity metrics are only meaningful observed-vs-observed. The theoretical
# spectrum has flat unit peak heights, so these are undefined/artefactual for the
# observed-vs-theoretical comparisons and are dropped there (kept only for query-vs-library).
_THEORETICAL_PREFIXES = ("q_obs__lib_theo__", "lib_obs__lib_theo__")
_OBS_VS_OBS_ONLY_SUFFIXES = ("cosine_similarity", "spectral_angle", "pearson_correlation")


def _skip_metric(short_key: str) -> bool:
    """True if this is an intensity-similarity metric on a theoretical comparison."""
    return short_key.startswith(_THEORETICAL_PREFIXES) and short_key.endswith(_OBS_VS_OBS_ONLY_SUFFIXES)

# Candidate locations for the shared Nature-methods palette (single source of truth).
_METADATA_COLORS_PATHS = [
    Path(__file__).resolve().parents[1]
    / "instanovo/foundational/eval/embed_eval_tasks/metadata_colors.json",
    Path("/home/hjisaac/Downloads/metadata_colors.json"),
]


def set_publication_style() -> None:
    """Apply the publication style shared with prior paper figures (publication.py)."""
    try:
        import seaborn as sns

        sns.set_theme(style="ticks")
    except ImportError:
        pass
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Palatino", "Palatino Linotype", "TeX Gyre Pagella",
                "Book Antiqua", "URW Palladio L", "DejaVu Serif",
            ],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.fontsize": 12,
            "legend.frameon": False,
            "legend.columnspacing": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
        }
    )


def load_shared_colors() -> Dict[str, str]:
    """Map the shared metadata palette to cross-set mirror semantics.

    ACFM=query, LCFM=library. The literal ``tier`` colors are near-identical pale
    blues (unreadable as thin mirror lines), so query/library use the saturated
    palette entries used by the existing rescue paper figures for consistency.
    """
    for path in _METADATA_COLORS_PATHS:
        if path.is_file():
            payload = json.loads(path.read_text())
            palette = payload.get("palette", [])
            pastel = payload.get("pastel", [])
            return {
                "query": palette[0] if palette else "#4E9AC6",       # ACFM query
                "library": palette[2] if len(palette) > 2 else "#F5A45D",  # LCFM library
                "matched_diff": palette[3] if len(palette) > 3 else "#C285C7",
                "unmatched": palette[7] if len(palette) > 7 else "#AAAAAA",
                "light_background": pastel[0] if pastel else "#EEF4FB",
                "dark_text": "#1A3F60",
            }
    return {
        "query": "#4E9AC6", "library": "#F5A45D", "matched_diff": "#C285C7",
        "unmatched": "#AAAAAA", "light_background": "#EEF4FB", "dark_text": "#1A3F60",
    }


def save_all_formats(fig: "plt.Figure", save_path: Path) -> None:
    """Save a figure as png/svg/pdf (paper deliverables) at 300 dpi."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(save_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the top-pairs explainer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--combined-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--rank", type=int, default=1, help="Which retrieval rank defines the pair (default rank-1)")
    parser.add_argument("--tolerance-da", type=float, default=0.05)
    parser.add_argument("--ion-types", default="by")
    parser.add_argument("--max-ion-charge", type=int, default=2)
    parser.add_argument("--id-key", default="usi")
    parser.add_argument("--peptide-key", default="library_peptide")
    parser.add_argument("--mz-key", default="mz_array")
    parser.add_argument("--intensity-key", default="intensity_array")
    parser.add_argument("--precursor-mz-key", default="precursor_mz")
    parser.add_argument("--precursor-charge-key", default="precursor_charge")
    parser.add_argument("--scan-key", default="scan")
    parser.add_argument("--rt-key", default="retention_time")
    return parser.parse_args()


def _scan_of(usi: Any) -> Optional[int]:
    match = re.search(r"scan=(\d+)", str(usi))
    return int(match.group(1)) if match else None


def _observed_dict(row: dict, args: argparse.Namespace, *, spectrum_id: str) -> dict:
    charge = row.get(args.precursor_charge_key)
    return {
        "mz": np.asarray(row[args.mz_key], dtype=float).flatten().tolist(),
        "intensity": np.asarray(row[args.intensity_key], dtype=float).flatten().tolist(),
        "spectrum_id": spectrum_id,
        "scan_number": _scan_of(spectrum_id),
        "precursor_mz": float(row[args.precursor_mz_key]) if row.get(args.precursor_mz_key) is not None else None,
        "precursor_charge": int(charge) if charge is not None else None,
        "rt_seconds": float(row[args.rt_key]) if row.get(args.rt_key) is not None else None,
        "source": "instanovo_cross_set",
    }


def _score_and_render(
    observed_dict: dict,
    peptide: str,
    *,
    candidate_extra: Dict[str, Any],
    out_dir: Path,
    mirror_name: str,
    annotated_name: str,
    args: argparse.Namespace,
) -> dict:
    """Score observed vs the peptide's theoretical spectrum and render mirror + annotated views."""
    try:
        from proteomics_mcp.core.scoring import score_candidate_spectrum
        from proteomics_mcp.core.visualization import (
            build_spectrum_payload,
            render_annotated_observed_spectrum_html,
            render_spectrum_comparison_html,
        )
        from proteomics_mcp.models.schemas import CandidatePSM, ObservedSpectrum
    except ImportError as exc:  # pragma: no cover - depends on an unpublished package
        raise ImportError(
            "This step scores observed spectra against theoretical ones, which needs "
            "`proteomics_mcp`. That package is unpublished and deliberately not included in this "
            "repository; see docs/sanitisation.md. Retrieval, rescue and the observed-versus-"
            "observed metrics do not need it and run without it."
        ) from exc

    observed = ObservedSpectrum(**observed_dict)
    charge = int(observed_dict.get("precursor_charge") or 2)
    predicted = build_predicted_spectrum(peptide, charge, ion_types=args.ion_types, max_ion_charge=args.max_ion_charge)
    candidate = CandidatePSM(
        candidate_id="cross_set",
        source_engine="instanovo_cross_set",
        peptidoform=peptide,
        scan_number=observed_dict.get("scan_number"),
        usi=observed_dict.get("spectrum_id"),
        spectrum_id=observed_dict.get("spectrum_id"),
        precursor_mz=observed_dict.get("precursor_mz"),
        precursor_charge=observed_dict.get("precursor_charge"),
        **candidate_extra,
    )
    result = score_candidate_spectrum(
        candidate=candidate,
        observed=observed,
        predicted=predicted,
        tolerance=args.tolerance_da,
        tolerance_unit="Da",
        annotation_ion_types=args.ion_types,
        annotation_max_ion_charge=args.max_ion_charge,
    )
    payload = build_spectrum_payload(observed, predicted, result)
    render_spectrum_comparison_html(payload, out_dir / mirror_name)
    render_annotated_observed_spectrum_html(
        payload,
        out_dir / annotated_name,
        peptidoform=peptide,
        ion_types=args.ion_types,
        tolerance=args.tolerance_da,
        max_ion_charge=args.max_ion_charge,
    )
    return result.model_dump()


def _pair_peaks(
    q_mz: np.ndarray, q_rel: np.ndarray, l_mz: np.ndarray, l_rel: np.ndarray, tolerance_da: float
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Greedy one-to-one peak matching -> per-peak diff rows plus matched masks.

    Each query peak is matched to at most one library peak within ``tolerance_da``
    (assigned in ascending |Δm/z| so the closest pairs win), giving a clean
    peak-level difference: matched, query-only, or library-only.
    """
    q_used = np.zeros(q_mz.size, dtype=bool)
    l_used = np.zeros(l_mz.size, dtype=bool)
    candidates: list[tuple[float, int, int]] = []
    if q_mz.size and l_mz.size:
        l_order = np.argsort(l_mz)
        l_sorted = l_mz[l_order]
        for qi, m in enumerate(q_mz):
            lo = int(np.searchsorted(l_sorted, m - tolerance_da, side="left"))
            hi = int(np.searchsorted(l_sorted, m + tolerance_da, side="right"))
            for k in range(lo, hi):
                lj = int(l_order[k])
                candidates.append((abs(float(l_mz[lj]) - float(m)), qi, lj))
    candidates.sort(key=lambda c: c[0])

    rows: list[dict] = []
    for _, qi, lj in candidates:
        if q_used[qi] or l_used[lj]:
            continue
        q_used[qi] = True
        l_used[lj] = True
        mzq, mzl = float(q_mz[qi]), float(l_mz[lj])
        rows.append({
            "status": "matched", "mz_query": mzq, "mz_library": mzl,
            "dmz_ppm": ((mzq - mzl) / mzl * 1e6) if mzl else None,
            "q_rel": float(q_rel[qi]), "l_rel": float(l_rel[lj]), "d_rel": float(q_rel[qi] - l_rel[lj]),
        })
    for qi in np.where(~q_used)[0].tolist():
        rows.append({
            "status": "query_only", "mz_query": float(q_mz[qi]), "mz_library": None, "dmz_ppm": None,
            "q_rel": float(q_rel[qi]), "l_rel": 0.0, "d_rel": float(q_rel[qi]),
        })
    for lj in np.where(~l_used)[0].tolist():
        rows.append({
            "status": "library_only", "mz_query": None, "mz_library": float(l_mz[lj]), "dmz_ppm": None,
            "q_rel": 0.0, "l_rel": float(l_rel[lj]), "d_rel": float(-l_rel[lj]),
        })
    rows.sort(key=lambda r: (r["mz_query"] if r["mz_query"] is not None else r["mz_library"]))
    return rows, q_used, l_used


def _plot_obs_vs_obs_mirror(
    q_obs: dict,
    l_obs: dict,
    *,
    peptide: str,
    block_a: dict,
    tolerance_da: float,
    save_path: Path,
    label: str,
) -> dict:
    """Mirror plot (query up / library down) + a peak-level difference-spectrum panel.

    Title is intentionally omitted from the canvas (encoded in the filename);
    legends carry the query/library/diff semantics. Writes ``peak_diff.csv`` and
    saves png/svg/pdf. Returns diff summary stats.
    """
    colors = load_shared_colors()
    blue = colors["query"]
    orange = colors["library"]
    purple = colors["matched_diff"]

    q_mz = np.asarray(q_obs["mz"], dtype=float)
    q_int = np.asarray(q_obs["intensity"], dtype=float)
    l_mz = np.asarray(l_obs["mz"], dtype=float)
    l_int = np.asarray(l_obs["intensity"], dtype=float)
    q_rel = q_int / (q_int.max() or 1.0)
    l_rel = l_int / (l_int.max() or 1.0)
    peak_rows, q_matched, _ = _pair_peaks(q_mz, q_rel, l_mz, l_rel, tolerance_da)

    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(10, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # --- Top: query (up) vs library (down); color = identity only ---
    ax.vlines(q_mz, 0, q_rel, color=blue, linewidth=0.9, zorder=2, label="Query")
    ax.vlines(l_mz, 0, -l_rel, color=orange, linewidth=0.9, zorder=2, label="Library")
    ax.axhline(0, color="#333333", linewidth=1.0)
    cos = block_a.get("cosine_similarity")
    matched_rate = (int(q_matched.sum()) / q_mz.size) if q_mz.size else 0.0
    ax.text(
        0.99, 0.95,
        f"cosine similarity = {_fmt(cos)}\n"
        f"matched peak rate = {matched_rate * 100:.1f}%  ({int(q_matched.sum())}/{q_mz.size})\n"
        f"match tolerance = ±{tolerance_da:g} Da",
        transform=ax.transAxes, va="top", ha="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": colors["light_background"], "edgecolor": "none"},
    )
    ax.set_ylabel("Relative intensity\n(query up / library down)")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(loc="upper left")

    # --- Bottom: peak-level difference (query − library) ---
    matched = [r for r in peak_rows if r["status"] == "matched"]
    q_only = [r for r in peak_rows if r["status"] == "query_only"]
    l_only = [r for r in peak_rows if r["status"] == "library_only"]
    if matched:
        ax_d.vlines([r["mz_query"] for r in matched], 0, [r["d_rel"] for r in matched],
                    color=purple, linewidth=1.0, zorder=3, label="Matched")
    if q_only:
        ax_d.vlines([r["mz_query"] for r in q_only], 0, [r["d_rel"] for r in q_only],
                    color=blue, linewidth=1.0, zorder=2, label="Query-only")
    if l_only:
        ax_d.vlines([r["mz_library"] for r in l_only], 0, [r["d_rel"] for r in l_only],
                    color=orange, linewidth=1.0, zorder=2, label="Library-only")
    ax_d.axhline(0, color="#333333", linewidth=1.0)
    ax_d.set_xlabel("m/z")
    ax_d.set_ylabel("Difference\n(query − library)")
    ax_d.set_ylim(-1.15, 1.15)
    ax_d.legend(loc="upper right", ncol=3)

    abs_drel = np.array([abs(r["d_rel"]) for r in matched], dtype=float)
    ppm = np.array([r["dmz_ppm"] for r in matched if r["dmz_ppm"] is not None], dtype=float)
    diff_stats = {
        "peak_diff_n_matched": len(matched),
        "peak_diff_n_query_only": len(q_only),
        "peak_diff_n_library_only": len(l_only),
        "peak_diff_median_abs_intensity_delta": float(np.median(abs_drel)) if abs_drel.size else None,
        "peak_diff_median_abs_mz_ppm": float(np.median(np.abs(ppm))) if ppm.size else None,
    }

    fig.tight_layout()
    save_all_formats(fig, save_path)
    plt.close(fig)

    pl.DataFrame(peak_rows).write_csv(save_path.parent / "peak_diff.csv")
    return diff_stats


def _fmt(value: Any, *, percent: bool = False, decimals: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html_lib.escape(str(value))
    if not np.isfinite(number):
        return "NA"
    if percent:
        return f"{number * 100:.1f}%"
    return f"{number:.{decimals}f}"


def _write_index(rows: list[dict], out_dir: Path, label: str) -> None:
    title = f"Top most-similar cross-set pairs{f' — {label}' if label else ''}"
    header = [
        "Pair", "Transferred peptide", "Query confidence", "Embedding cosine", "Embedding margin", "Query scan", "Lib scan", "Scan gap",
        "Query-obs vs Library-obs: cosine", "Query-obs vs Library-obs: matched peak frac (of query)",
        "Query-obs vs Peptide-theo: matched ion frac", "Query-obs vs Peptide-theo: explained intensity",
        "Query-obs vs Peptide-theo: hyperscore", "Query-obs vs Peptide-theo: residue coverage",
        "Query-obs vs Peptide-theo: consecutive ions", "Query-obs vs Peptide-theo: TIC explained",
        "Library-obs vs Peptide-theo: matched ion frac", "Library-obs vs Peptide-theo: residue coverage", "Views",
    ]
    body_rows = []
    for r in rows:
        views = " | ".join(
            f'<a href="{r["dir"]}/{fname}">{lbl}</a>'
            for lbl, fname in [
                ("query↔library obs (PNG)", "query_vs_library_observed_spectra_peak_difference.png"),
                ("query↔peptide", "query_vs_peptide_mirror.html"),
                ("query annot.", "query_annotated.html"),
                ("lib↔peptide", "library_vs_peptide_mirror.html"),
                ("lib annot.", "library_annotated.html"),
            ]
        )
        cells = [
            r["pair"], html_lib.escape(r["transferred_peptide"]),
            _fmt(r.get("query_spectrum_confidence")),
            _fmt(r["embedding_cosine_similarity"]), _fmt(r["embedding_margin_rank1_minus_rank2"]),
            str(r["query_scan"]), str(r["library_scan"]), str(r["scan_gap"]),
            _fmt(r[f"{CMP_A}__cosine_similarity"]),
            _fmt(r.get(f"{CMP_A}__matched_peak_fraction_of_query"), percent=True),
            _fmt(r[f"{CMP_B}__matched_ion_fraction"], percent=True),
            _fmt(r[f"{CMP_B}__explained_intensity_fraction"], percent=True),
            _fmt(r[f"{CMP_B}__hyperscore"]),
            _fmt(r.get(f"{CMP_B}__residue_evidence_coverage"), percent=True),
            str(r.get(f"{CMP_B}__consecutive_ion_series")),
            _fmt(r.get(f"{CMP_B}__tic_explained"), percent=True),
            _fmt(r[f"{CMP_C}__matched_ion_fraction"], percent=True),
            _fmt(r.get(f"{CMP_C}__residue_evidence_coverage"), percent=True), views,
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html_lib.escape(title)}</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", sans-serif; color:#27313f; margin:24px; }}
 h1 {{ font-size:20px; }}
 table {{ border-collapse: collapse; font-size:13px; }}
 th,td {{ border:1px solid #D8E0EA; padding:6px 9px; text-align:right; }}
 th {{ background:#F8FAFC; }} td:nth-child(2) {{ font-family:monospace; text-align:left; }}
 td:last-child {{ text-align:left; }} a {{ color:#068D9D; }}
 caption {{ text-align:left; color:#64748B; font-size:12px; margin-bottom:8px; }}
</style></head><body>
<h1>{html_lib.escape(title)}</h1>
<table><caption><b>Query-obs vs Library-obs</b> = the two real spectra compared directly (spectral similarity). <b>Query-obs vs Peptide-theo</b> = query spectrum vs the transferred peptide's theoretical fragment ions (does the peptide explain the query?). <b>Library-obs vs Peptide-theo</b> = library spectrum vs the same peptide (self-consistency reference).
<b>residue coverage</b> = fraction of peptide backbone bracketed by matched b/y ions (best "is the sequence supported" signal); <b>consecutive ions</b> = longest unbroken ion ladder; <b>TIC explained</b> = fraction of total ion current explained by matched ions. Query values should approach the library self-reference if the transfer is chemically valid. Every computed metric is in top_pairs_summary.csv with fully explicit column names.</caption>
<tr>{"".join(f"<th>{h}</th>" for h in header)}</tr>
{"".join(body_rows)}
</table></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    """Render match-quality explanation views for the top-N most-similar pairs."""
    args = parse_args()
    if not MCP_AVAILABLE:
        raise SystemExit("proteomics-mcp is required. Install with: uv sync --extra proteomics-metrics")

    set_publication_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pl.read_csv(args.candidates_csv)
    pairs = (
        candidates.filter(pl.col("rank") == args.rank)
        .sort("embedding_score", descending=True)
        .head(args.top_n)
        .to_dicts()
    )

    ids = {str(p["query_id"]) for p in pairs} | {str(p["library_id"]) for p in pairs}
    columns = list(
        dict.fromkeys(
            [
                args.id_key, args.mz_key, args.intensity_key, args.precursor_mz_key,
                args.precursor_charge_key, args.scan_key, args.rt_key, "sequence", "unmodified_peptide",
            ]
        )
    )
    spectra = pl.read_parquet(args.combined_parquet, columns=columns).filter(pl.col(args.id_key).is_in(list(ids)))
    by_id = {str(row[args.id_key]): row for row in spectra.to_dicts()}

    summary_rows: list[dict] = []
    for i, pair in enumerate(pairs, 1):
        query_id = str(pair["query_id"])
        library_id = str(pair["library_id"])
        peptide = str(pair.get(args.peptide_key) or "")
        q_row = by_id.get(query_id)
        l_row = by_id.get(library_id)
        if q_row is None or l_row is None or not peptide:
            print(f"[pair {i}] skipped (missing spectrum or peptide)")
            continue

        pair_dir = out_dir / f"pair_{i:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        q_obs = _observed_dict(q_row, args, spectrum_id=query_id)
        l_obs = _observed_dict(l_row, args, spectrum_id=library_id)

        # Canonical Stage-2 metrics (blocks A/B/C + lens) via the worker, so the
        # sequence-evidence lens metrics carry the exact same prefixes/semantics.
        lib_unmod = str(l_row.get("unmodified_peptide") or peptide)
        q_rec = SpectrumRecord(
            mz=tuple(q_obs["mz"]), intensity=tuple(q_obs["intensity"]),
            precursor_mz=q_obs["precursor_mz"], precursor_charge=q_obs["precursor_charge"],
            peptide="", unmodified_peptide="",
        )
        l_rec = SpectrumRecord(
            mz=tuple(l_obs["mz"]), intensity=tuple(l_obs["intensity"]),
            precursor_mz=l_obs["precursor_mz"], precursor_charge=l_obs["precursor_charge"],
            peptide=peptide, unmodified_peptide=lib_unmod,
        )
        _, block_c_metrics = score_library_self(
            LibrarySelfWorkItem(library_index=0, library=l_rec, tolerance_da=args.tolerance_da,
                                ion_types=args.ion_types, max_ion_charge=args.max_ion_charge)
        )
        worker_metrics = score_query_rank(
            QueryRankWorkItem(query_index=0, rank=1, library_index=0, query=q_rec, library=l_rec,
                              score_blocks=("A", "B"), tolerance_da=args.tolerance_da,
                              ion_types=args.ion_types, max_ion_charge=args.max_ion_charge),
            library_self_cache={0: block_c_metrics},
        )

        _score_and_render(
            q_obs,
            peptide,
            candidate_extra={
                "raw_score": float(pair.get("embedding_score")) if pair.get("embedding_score") is not None else None,
                "raw_score_name": "embedding cosine",
                "next_beam_margin": float(pair.get("embedding_margin_1_2")) if pair.get("embedding_margin_1_2") is not None else None,
                "confidence": float(pair.get("query_spectrum_confidence")) if pair.get("query_spectrum_confidence") is not None else None,
            },
            out_dir=pair_dir,
            mirror_name="query_vs_peptide_mirror.html",
            annotated_name="query_annotated.html",
            args=args,
        )
        _score_and_render(
            l_obs,
            peptide,
            candidate_extra={"raw_score_name": "library self"},
            out_dir=pair_dir,
            mirror_name="library_vs_peptide_mirror.html",
            annotated_name="library_annotated.html",
            args=args,
        )
        block_a = {
            "cosine_similarity": worker_metrics.get("q_obs__lib_obs__cosine_similarity"),
            "matched_peak_count": worker_metrics.get("q_obs__lib_obs__matched_peak_count"),
        }
        diff_stats = _plot_obs_vs_obs_mirror(
            q_obs,
            l_obs,
            peptide=peptide,
            block_a=block_a,
            tolerance_da=args.tolerance_da,
            save_path=pair_dir / "query_vs_library_observed_spectra_peak_difference",
            label=f"{args.label} pair {i}".strip(),
        )

        q_scan = _scan_of(query_id)
        l_scan = _scan_of(library_id)
        # Peak counts and matched-peak fractions for the observed-vs-observed comparison.
        n_query_peaks = int(np.asarray(q_obs["mz"]).size)
        n_library_peaks = int(np.asarray(l_obs["mz"]).size)
        matched_peak_count = worker_metrics.get("q_obs__lib_obs__matched_peak_count")
        matched_frac_query = (matched_peak_count / n_query_peaks) if (matched_peak_count is not None and n_query_peaks) else None
        matched_frac_library = (matched_peak_count / n_library_peaks) if (matched_peak_count is not None and n_library_peaks) else None
        n_matched = diff_stats["peak_diff_n_matched"]
        n_query_only = diff_stats["peak_diff_n_query_only"]
        n_library_only = diff_stats["peak_diff_n_library_only"]

        # Curated, fully explicit headline metrics up front. Observed-vs-observed (the two
        # real spectra) is prioritised first, then the observed-vs-theoretical comparisons.
        # CMP_A/B/C keys equal the explicit-renamed raw keys below, so the full-dump loop
        # does not duplicate them.
        row = {
            "pair": i,
            "dir": pair_dir.name,
            "query_id": query_id,
            "library_id": library_id,
            "transferred_peptide": peptide,
            # Query-level model confidence (identification-free spectrum_confidence used to
            # pre-select the confident queries); constant across a query's ranks.
            "query_spectrum_confidence": pair.get("query_spectrum_confidence"),
            "embedding_cosine_similarity": pair.get("embedding_score"),
            "embedding_margin_rank1_minus_rank2": pair.get("embedding_margin_1_2"),
            "query_scan": q_scan,
            "library_scan": l_scan,
            "scan_gap": (abs(q_scan - l_scan) if q_scan is not None and l_scan is not None else None),
            # --- Observed vs observed (query vs library) --- prioritised first ---
            f"{CMP_A}__cosine_similarity": worker_metrics.get("q_obs__lib_obs__cosine_similarity"),
            f"{CMP_A}__spectral_angle": worker_metrics.get("q_obs__lib_obs__spectral_angle"),
            f"{CMP_A}__pearson_correlation": worker_metrics.get("q_obs__lib_obs__pearson_correlation"),
            f"{CMP_A}__query_peak_count": n_query_peaks,
            f"{CMP_A}__library_peak_count": n_library_peaks,
            f"{CMP_A}__matched_peak_count": matched_peak_count,
            f"{CMP_A}__matched_peak_fraction_of_query": matched_frac_query,
            f"{CMP_A}__matched_peak_fraction_of_library": matched_frac_library,
            f"{CMP_A}__explained_intensity_fraction": worker_metrics.get("q_obs__lib_obs__explained_intensity_fraction"),
            # peak-level diff (also observed vs observed)
            "peak_diff_n_matched": n_matched,
            "peak_diff_n_query_only": n_query_only,
            "peak_diff_n_library_only": n_library_only,
            "peak_diff_matched_fraction_of_query": (n_matched / (n_matched + n_query_only)) if (n_matched + n_query_only) else None,
            "peak_diff_matched_fraction_of_library": (n_matched / (n_matched + n_library_only)) if (n_matched + n_library_only) else None,
            "peak_diff_median_abs_intensity_delta": diff_stats["peak_diff_median_abs_intensity_delta"],
            "peak_diff_median_abs_mz_ppm": diff_stats["peak_diff_median_abs_mz_ppm"],
            # --- Query observed vs transferred-peptide theoretical ---
            f"{CMP_B}__matched_ion_fraction": worker_metrics.get("q_obs__lib_theo__matched_ion_fraction"),
            f"{CMP_B}__explained_intensity_fraction": worker_metrics.get("q_obs__lib_theo__explained_intensity_fraction"),
            f"{CMP_B}__hyperscore": worker_metrics.get("q_obs__lib_theo__hyperscore"),
            f"{CMP_B}__matched_ion_count": worker_metrics.get("q_obs__lib_theo__matched_ion_count"),
            f"{CMP_B}__residue_evidence_coverage": worker_metrics.get("q_obs__lib_theo__residue_evidence_coverage"),
            f"{CMP_B}__consecutive_ion_series": worker_metrics.get("q_obs__lib_theo__consecutive_ion_series"),
            f"{CMP_B}__tic_explained": worker_metrics.get("q_obs__lib_theo__tic_explained"),
            # --- Library observed vs same peptide theoretical (self-consistency) ---
            f"{CMP_C}__matched_ion_fraction": block_c_metrics.get("lib_obs__lib_theo__matched_ion_fraction"),
            f"{CMP_C}__explained_intensity_fraction": block_c_metrics.get("lib_obs__lib_theo__explained_intensity_fraction"),
            f"{CMP_C}__residue_evidence_coverage": block_c_metrics.get("lib_obs__lib_theo__residue_evidence_coverage"),
        }
        # Keep EVERY computed metric with explicit names (no block A/B/C shorthand), so
        # nothing helpful is dropped from the results table.
        for key, value in worker_metrics.items():
            if key in ("query_index", "rank", "library_index") or _skip_metric(key):
                continue
            row.setdefault(_explicit_key(key), value)
        for key, value in block_c_metrics.items():
            if _skip_metric(key):
                continue
            row.setdefault(_explicit_key(key), value)
        summary_rows.append(row)
        print(f"[pair {i}] {peptide}  embed={_fmt(pair.get('embedding_score'))}  "
              f"query_vs_library_cosine={_fmt(row[f'{CMP_A}__cosine_similarity'])}  "
              f"query_vs_peptide_matched_ion_frac={_fmt(row[f'{CMP_B}__matched_ion_fraction'], percent=True)}  "
              f"residue_coverage(query)={_fmt(row[f'{CMP_B}__residue_evidence_coverage'], percent=True)}  "
              f"consecutive_ions(query)={row[f'{CMP_B}__consecutive_ion_series']}")

    pl.DataFrame(summary_rows).write_csv(out_dir / "top_pairs_summary.csv")
    (out_dir / "top_pairs_summary.json").write_text(json.dumps(summary_rows, indent=2, default=str))
    _write_index(summary_rows, out_dir, args.label)
    print(f"\nWrote {len(summary_rows)} pairs + index.html to {out_dir}")


if __name__ == "__main__":
    main()
