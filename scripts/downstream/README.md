# Downstream retrieval, rescue and run classification

These scripts support the downstream applications — database-free identification, spectral rescue
and run-level classification — rather than corpus construction. The three stages described in
[`../README.md`](../README.md) build the training corpora; nothing here does.

Unlike the pipeline stages, these take **argparse** arguments, not Typer. `--help` on any of them
is authoritative.

## Building the datasets

The evaluation tasks do not consume raw spectra. They need anchor and query roles attached, so the
dataset is built first:

| Script | Builds |
|---|---|
| `create_cross_set_annotation_transfer_dataset.py` | the cross-set query/anchor parquet |
| `create_spectral_rescue_reformulated_dataset.py` | the reformulated rescue dataset |
| `create_sequence_rescue_dataset.py` | the sequence-rescue dataset |

## Choosing what to score

Evidence scoring is the expensive step, so which queries reach it is chosen deliberately:

| Script | Role |
|---|---|
| `scan_spectral_rescue_candidates_light.py` | fast pre-screen for projects and base/modified pairs with enough spectra |
| `discover_spectral_rescue_pairs.py` | selects the base/modified pairs for a run |
| `sample_query_pool.py` | samples a rank-1 query pool, so selectors compete on the same queries |
| `select_topk_queries.py` | query-side top-K pruning before stage 2 |
| `select_queries_margin_analysis.py` | margin-based query selection |
| `compare_query_selectors.py` | head-to-head comparison of the selectors above |

## Running the evaluation

Retrieval itself runs through the ordinary evaluation entry point:

```bash
instanovo-fm evaluate \
    --checkpoint path/to/model_best.ckpt \
    evaluation.tasks_to_run=[crosssetannotationtransfertask]
```

Cross-set retrieval is split in two on purpose. The evaluator does the retrieval and writes
`cross_set_topk_candidates.csv`; `compute_cross_set_evidence_metrics.py` then scores the evidence
blocks offline from that file. The second stage carries the cost, so it is resumable and kept
separate.

> **The evidence blocks need a package that is not shipped here**
>
> Blocks B and C score a query against the *theoretical* spectrum of its transferred peptide, which
> needs `proteomics-mcp` — unpublished work by its author, and deliberately excluded. See
> [Sanitisation](../../docs/sanitisation.md#what-is-not-ported). Retrieval, rescue and the
> observed-versus-observed metrics do not need it.

## Figures

`plot_spectral_rescue_publication.py`, `plot_sequence_rescue_publication.py`,
`plot_kostas_preprint_figures.py`, `plot_crossset_diagnostics.py`, `plot_margin_evidence.py`,
`plot_rescue_projection_comparison.py` and `regenerate_spectral_rescue_plots.py` draw from saved
artefacts, so they can be re-run without repeating the evaluation.

Colours come from [`config/metadata_colors.json`](../../config/metadata_colors.json), the palette
shared with the paper figures.
