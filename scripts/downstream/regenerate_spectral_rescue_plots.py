#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Regenerate reformulated spectral-rescue publication plots locally.

UMAP neighborhood plots are built from the saved query×library similarity matrix
(``rescue_similarity_matrix.npz``), so they can be redone from a completed run
without re-embedding. Spectrum overlay plots additionally need ``embeddings.h5``.

Examples:
    # UMAP + distribution plots from an existing AIchor seed directory
    uv run python scripts/downstream/regenerate_spectral_rescue_plots.py \\
        --artifact-dir instanovo/foundational/eval/embed_eval_results/spectral_rescue_rigorous/seed_42

    # UMAP only (fast iteration on styling)
    uv run python scripts/downstream/regenerate_spectral_rescue_plots.py \\
        --artifact-dir .../seed_42 --umap-only

    # All plots including spectrum overlays (needs embeddings.h5 in the same dir)
    uv run python scripts/downstream/regenerate_spectral_rescue_plots.py \\
        --artifact-dir .../seed_42 --embeddings-dir .../seed_42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated_plots import (
    load_rescue_artifacts_from_dir,
    save_rescue_publication_plots,
)


def _resolve_output_dir(artifact_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    task_subdir = artifact_dir / "spectralrescuetaskreformulated"
    if (task_subdir / "rescue_similarity_matrix.npz").is_file():
        return task_subdir
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Seed output directory from a prior rescue run (contains npz/csv artifacts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: directory containing the npz artifacts)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory with embeddings.h5 for spectrum overlay plots (default: artifact-dir)",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--plot-max-pair-scores", type=int, default=50_000)
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Regenerate only the three UMAP neighborhood figures",
    )
    parser.add_argument(
        "--skip-distributions",
        action="store_true",
        help="Skip margin / match-unmatch distribution plots",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    out_dir = _resolve_output_dir(artifact_dir, args.output_dir.resolve() if args.output_dir else None)
    out_dir.mkdir(parents=True, exist_ok=True)

    S, selection, query_metrics, meta = load_rescue_artifacts_from_dir(
        artifact_dir,
        embeddings_dir=args.embeddings_dir,
    )

    task = SpectralRescueTaskReformulated(sample_seed=args.sample_seed)
    paths = save_rescue_publication_plots(
        out_dir,
        S,
        selection,
        query_metrics,
        positive_library_role=task.positive_library_role,
        negative_library_role=task.negative_library_role,
        modified_query_role=task.modified_query_role,
        min_negative_clean_edit_distance=task.min_negative_clean_edit_distance,
        clean_edit_distance_fn=task._clean_site_edit_distance,
        sample_seed=args.sample_seed,
        plot_dpi=args.plot_dpi,
        plot_max_pair_scores=args.plot_max_pair_scores,
        meta=meta,
        include_distribution_plots=not args.skip_distributions and not args.umap_only,
        include_umap_plots=True,
        include_spectrum_plots=not args.umap_only,
    )

    summary_path = out_dir / "regenerated_plot_paths.json"
    summary_path.write_text(json.dumps(paths, indent=2))
    print(f"Wrote {len(paths)} plots to {out_dir}")
    for key, path in sorted(paths.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
