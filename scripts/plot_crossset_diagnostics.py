# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Cross-set (no-query-ground-truth) diagnostic plots for annotation transfer.

Because the query spectra are unannotated, we cannot label library spectra as
"same sequence as the query" (that requires ground truth we do not have). This
script therefore produces the *honest* analogues for a single run:

1. Decoy / null plot  -- chemical fit (matched-ion fraction, hyperscore) of the
   transferred rank-1 peptide vs a randomly assigned library peptide. This is the
   honest stand-in for the duplicate-rescue "match vs unmatched" separation: it
   uses chemistry instead of ground-truth labels.
2. Distribution plots  -- (a) rank-1 embedding cosine and (b) rank-1
   query-vs-library observed-spectrum cosine across all queries, showing how
   confident/consistent the transferred matches are overall.

Everything here needs only the spectra parquet, the rank-1 evidence CSV, and the
full candidates CSV -- no embeddings or GPU. (The hero-query UMAP is a separate
script because it requires the raw embedding matrix.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from instanovo_fm.eval.spectrum_metrics.mcp_scoring import (
    MCP_AVAILABLE,
    score_observed_vs_theoretical,
)
from instanovo_fm.eval.spectrum_metrics.observed_vs_observed import (
    score_observed_vs_observed,
)

# Reuse the shared publication style / palette / multi-format saver.
from scripts.explain_top_pairs import (  # noqa: E402
    load_shared_colors,
    save_all_formats,
    set_publication_style,
)


def _log(t0: float, msg: str) -> None:
    print(f"[t={time.time() - t0:7.1f}s] {msg}", flush=True)


def _query_records(df: pl.DataFrame) -> dict[str, dict]:
    """Map query USI -> observed spectrum payload for the ACFM (query) tier."""
    q = df.filter(pl.col("search_tier") == "acfm")
    records: dict[str, dict] = {}
    for row in q.iter_rows(named=True):
        records[row["usi"]] = {
            "mz": np.asarray(row["mz_array"], dtype=float),
            "intensity": np.asarray(row["intensity_array"], dtype=float),
            "precursor_mz": row["precursor_mz"],
            "precursor_charge": row["precursor_charge"],
        }
    return records


def _library_records(df: pl.DataFrame) -> dict[str, dict]:
    """Map library USI -> observed spectrum payload for the LCFM (library) tier."""
    lib = df.filter(pl.col("search_tier") == "lcfm")
    records: dict[str, dict] = {}
    for row in lib.iter_rows(named=True):
        records[row["usi"]] = {
            "mz": np.asarray(row["mz_array"], dtype=float),
            "intensity": np.asarray(row["intensity_array"], dtype=float),
        }
    return records


# --------------------------------------------------------------------------- #
# Plot 1: decoy / null                                                        #
# --------------------------------------------------------------------------- #
def plot_decoy(
    evidence_csv: Path,
    parquet: Path,
    out_dir: Path,
    *,
    t0: float,
    seed: int = 0,
    tolerance_da: float = 0.05,
) -> None:
    """Transferred-peptide fit vs a random library peptide (chemical decoy)."""
    if not MCP_AVAILABLE:
        _log(t0, "proteomics-mcp unavailable; skipping decoy plot.")
        return

    ev = pl.read_csv(evidence_csv)
    df = pl.read_parquet(parquet)
    q_rec = _query_records(df)

    # Pool of real library peptides to draw decoys from.
    lib_peptides = (
        df.filter(pl.col("search_tier") == "lcfm")
        .select("sequence")
        .filter(pl.col("sequence").is_not_null() & (pl.col("sequence") != ""))
        .to_series()
        .unique()
        .to_list()
    )
    rng = np.random.default_rng(seed)

    metric = "matched_ion_fraction"
    real_col = f"q_obs__lib_theo__{metric}"
    real_hyper = "q_obs__lib_theo__hyperscore"

    real_fit: list[float] = []
    decoy_fit: list[float] = []
    real_hyp: list[float] = []
    decoy_hyp: list[float] = []

    n = ev.height
    for i, row in enumerate(ev.iter_rows(named=True)):
        qid = row["query_id"]
        true_pep = row["library_peptide"]
        rec = q_rec.get(qid)
        if rec is None or rec["mz"].size == 0:
            continue
        # real arm (already computed upstream)
        if row.get(real_col) is not None:
            real_fit.append(float(row[real_col]))
        if row.get(real_hyper) is not None:
            real_hyp.append(float(row[real_hyper]))
        # decoy arm: random library peptide != the transferred one
        decoy_pep = str(rng.choice(lib_peptides))
        for _ in range(4):
            if decoy_pep != true_pep:
                break
            decoy_pep = str(rng.choice(lib_peptides))
        try:
            b = score_observed_vs_theoretical(
                observed_mz=rec["mz"],
                observed_intensity=rec["intensity"],
                peptidoform=decoy_pep,
                precursor_mz=rec["precursor_mz"],
                precursor_charge=rec["precursor_charge"],
                tolerance_da=tolerance_da,
            )
        except Exception:
            b = {}
        if b.get(metric) is not None:
            decoy_fit.append(float(b[metric]))
        if b.get("hyperscore") is not None:
            decoy_hyp.append(float(b["hyperscore"]))
        if (i + 1) % 200 == 0:
            _log(t0, f"decoy scored {i + 1}/{n}")

    colors = load_shared_colors()
    c_real, c_decoy = colors["query"], colors["unmatched"]

    # matched-ion fraction
    _decoy_panel(
        real_fit,
        decoy_fit,
        xlabel="Matched-ion fraction (query peaks explained by peptide)",
        save_path=out_dir / "transferred_vs_random_peptide_matched_ion_fraction",
        c_real=c_real,
        c_decoy=c_decoy,
        bins=np.linspace(0, 1, 26),
    )
    # hyperscore
    if real_hyp and decoy_hyp:
        lo = min(min(real_hyp), min(decoy_hyp))
        hi = max(max(real_hyp), max(decoy_hyp))
        _decoy_panel(
            real_hyp,
            decoy_hyp,
            xlabel="Hyperscore (query vs peptide theoretical spectrum)",
            save_path=out_dir / "transferred_vs_random_peptide_hyperscore",
            c_real=c_real,
            c_decoy=c_decoy,
            bins=np.linspace(lo, hi, 30),
        )
    _log(
        t0,
        f"decoy: real n={len(real_fit)} median={np.median(real_fit):.3f} | "
        f"random n={len(decoy_fit)} median={np.median(decoy_fit):.3f}",
    )


def _decoy_panel(
    real: list[float],
    decoy: list[float],
    *,
    xlabel: str,
    save_path: Path,
    c_real: str,
    c_decoy: str,
    bins: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(
        decoy, bins=bins, density=True, color=c_decoy, alpha=0.55,
        label=f"Random peptide (decoy), median={np.median(decoy):.3f}",
    )
    ax.hist(
        real, bins=bins, density=True, color=c_real, alpha=0.75,
        label=f"Transferred peptide (rank-1), median={np.median(real):.3f}",
    )
    ax.axvline(float(np.median(decoy)), color=c_decoy, ls="--", lw=1.5)
    ax.axvline(float(np.median(real)), color=c_real, ls="--", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_all_formats(fig, save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plot 2 + 3: distributions across queries                                    #
# --------------------------------------------------------------------------- #
def plot_distributions(
    candidates_csv: Path,
    parquet: Path,
    out_dir: Path,
    *,
    t0: float,
    max_obs_cosine: Optional[int],
    tolerance_da: float = 0.05,
) -> None:
    colors = load_shared_colors()
    c = colors["query"]

    _log(t0, "loading rank-1 candidates ...")
    rank1 = (
        pl.scan_csv(candidates_csv)
        .filter(pl.col("rank") == 1)
        .select(["query_id", "library_id", "embedding_score"])
        .collect()
    )
    _log(t0, f"rank-1 queries: {rank1.height}")

    # (a) embedding cosine over all queries
    emb = rank1["embedding_score"].drop_nulls().to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(emb, bins=np.linspace(float(emb.min()), 1.0, 40), color=c, alpha=0.8)
    ax.axvline(float(np.median(emb)), color=colors["dark_text"], ls="--", lw=1.5,
               label=f"median = {np.median(emb):.3f}")
    ax.set_xlabel("Rank-1 embedding cosine similarity (query vs best library match)")
    ax.set_ylabel("Number of queries")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_all_formats(fig, out_dir / "rank1_embedding_cosine_distribution")
    plt.close(fig)
    _log(t0, f"embedding cosine: n={emb.size} median={np.median(emb):.3f}")

    # (b) observed-vs-observed cosine over (a sample of) rank-1 pairs
    df = pl.read_parquet(parquet)
    q_rec = _query_records(df)
    l_rec = _library_records(df)

    pairs = rank1
    if max_obs_cosine is not None and rank1.height > max_obs_cosine:
        pairs = rank1.sample(n=max_obs_cosine, seed=0)
        _log(t0, f"sampling {max_obs_cosine} pairs for observed-vs-observed cosine")

    obs_cos: list[float] = []
    total = pairs.height
    for i, row in enumerate(pairs.iter_rows(named=True)):
        q = q_rec.get(row["query_id"])
        lib = l_rec.get(row["library_id"])
        if q is None or lib is None or q["mz"].size == 0 or lib["mz"].size == 0:
            continue
        a = score_observed_vs_observed(
            q["mz"], q["intensity"], lib["mz"], lib["intensity"], tolerance_da=tolerance_da
        )
        if a.get("cosine_similarity") is not None:
            obs_cos.append(float(a["cosine_similarity"]))
        if (i + 1) % 2000 == 0:
            _log(t0, f"obs-obs cosine {i + 1}/{total}")

    obs = np.asarray(obs_cos, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(obs, bins=np.linspace(0, 1, 40), color=colors["library"], alpha=0.8)
    ax.axvline(float(np.median(obs)), color=colors["dark_text"], ls="--", lw=1.5,
               label=f"median = {np.median(obs):.3f}")
    ax.set_xlabel("Rank-1 observed-spectrum cosine similarity (query vs library, binned peaks)")
    ax.set_ylabel("Number of queries")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_all_formats(fig, out_dir / "rank1_observed_spectrum_cosine_distribution")
    plt.close(fig)
    _log(t0, f"observed cosine: n={obs.size} median={np.median(obs):.3f}")


# --------------------------------------------------------------------------- #
# Plot 4: hero-query UMAP (requires raw embeddings)                            #
# --------------------------------------------------------------------------- #
def plot_hero_umap(
    embeddings_npz: Path,
    out_dir: Path,
    *,
    t0: float,
    num_queries: int = 5,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    seed: int = 42,
) -> None:
    """Honest hero-query UMAP: a few query stars over the full library map.

    No query ground truth exists, so a query dot is only paired with its *rank-1
    transferred* peptide's library spectra (the peptide we would assign), never
    "same sequence as the query". With ~84% singleton library peptides and near
    degenerate embeddings (rank-1 cosine ~0.99), expect a single dense blob rather
    than clean clusters -- that itself is the honest visual message.
    """
    try:
        import umap  # noqa: F401
    except ImportError:
        _log(t0, "umap-learn not installed (uv sync --extra <umap group>); skipping UMAP.")
        return
    import umap as umap_mod

    data = np.load(embeddings_npz, allow_pickle=True)
    e_q = data["query_embeddings"].astype(np.float32)
    e_lib = data["library_embeddings"].astype(np.float32)
    lib_peptides = data["library_peptides"].astype(object)
    q_top1 = data["query_top1_peptides"].astype(object)

    n_q, n_lib = e_q.shape[0], e_lib.shape[0]
    if n_q == 0 or n_lib == 0:
        _log(t0, "empty embeddings; skipping UMAP.")
        return

    rng = np.random.RandomState(seed)
    n_hero = min(num_queries, n_q)
    hero_rows = np.sort(rng.choice(n_q, n_hero, replace=False))
    hero_peptides = [str(q_top1[r]) for r in hero_rows]

    combined = np.concatenate([e_lib, e_q[hero_rows]], axis=0)
    reducer = umap_mod.UMAP(
        n_neighbors=min(n_neighbors, max(2, combined.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    coords = np.asarray(reducer.fit_transform(combined))
    lib_xy, q_xy = coords[:n_lib], coords[n_lib:]

    colors = load_shared_colors()
    fig, ax = plt.subplots(figsize=(9, 7))
    # Whole library = neutral grey backdrop (the "map").
    ax.scatter(lib_xy[:, 0], lib_xy[:, 1], s=18, c=colors["unmatched"], alpha=0.5,
               linewidths=0, zorder=1, label=f"Library spectra (n={n_lib})")

    cmap = plt.cm.get_cmap("tab10", max(n_hero, 1))
    for i, (_r, pep) in enumerate(zip(hero_rows.tolist(), hero_peptides)):
        col = cmap(i % 10)
        # Highlight the transferred peptide's library spectra (its rank-1 anchor + siblings).
        sib = [j for j, lp in enumerate(lib_peptides) if pep and str(lp) == pep]
        if sib:
            ax.scatter(lib_xy[sib, 0], lib_xy[sib, 1], s=70, c=[col], alpha=0.9,
                       edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(q_xy[i, 0], q_xy[i, 1], s=320, marker="*", c=[col],
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label=f"Q{i + 1} \u2192 {pep[:18]}" + ("\u2026" if len(pep) > 18 else ""))

    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    save_all_formats(fig, out_dir / "hero_query_umap_over_library")
    plt.close(fig)
    _log(t0, f"UMAP: {n_hero} hero queries over {n_lib} library spectra")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-csv", type=Path,
                    default=Path("stage2_477_1/blockB_rank1/cross_set_topk_candidates_with_evidence.csv"))
    ap.add_argument("--candidates-csv", type=Path,
                    default=Path("stage2_477_1/cross_set_topk_candidates.csv"))
    ap.add_argument("--parquet", type=Path,
                    default=Path("stage2_477_1/PXD074343_477-1_all.parquet"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("stage2_477_1/plots/crossset_diagnostics"))
    ap.add_argument("--max-obs-cosine", type=int, default=None,
                    help="Sample this many rank-1 pairs for observed cosine (default: all).")
    ap.add_argument("--embeddings-npz", type=Path, default=None,
                    help="cross_set_embeddings.npz from a Stage-1 run with "
                         "save_embeddings_artifact=true; enables the hero-query UMAP.")
    ap.add_argument("--skip-decoy", action="store_true")
    ap.add_argument("--skip-distributions", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    t0 = time.time()

    if not args.skip_decoy:
        plot_decoy(args.evidence_csv, args.parquet, args.out_dir, t0=t0)
    if not args.skip_distributions:
        plot_distributions(
            args.candidates_csv, args.parquet, args.out_dir,
            t0=t0, max_obs_cosine=args.max_obs_cosine,
        )
    if args.embeddings_npz is not None and args.embeddings_npz.is_file():
        plot_hero_umap(args.embeddings_npz, args.out_dir, t0=t0)
    elif args.embeddings_npz is not None:
        _log(t0, f"embeddings npz not found: {args.embeddings_npz}")
    _log(t0, f"done. outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
