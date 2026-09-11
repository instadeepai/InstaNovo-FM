# How-to: Reproduce the Foundation Model results

Every result in the paper comes from a config in `src/instanovo_fm/configs/` and a CLI invocation that
applies it. This page gives both.

For what the model is and how it works, see
[The InstaNovo Foundation Model](foundation_model.md). For training it, see
[Training the Foundation Model](foundation_model_training.md).

## The published model

**LCFM TS·noPA** — 12 layers, d=768, 12 heads, feed-forward 3072, ~89.5 M parameters, trained for
230,000 steps on the LCFM corpus.

`configs/model/foundation_base.yaml` is the *ablation* baseline, at 9 layers and feed-forward 1024.
The published model is that baseline plus three overrides:

```bash
instanovo-fm train \
    model=foundation_base \
    dataset=lcfm \
    model.n_layers=12 \
    model.dim_feedforward=3072 \
    training_steps=230000
```

A bare `foundation_base` run does not reproduce the paper.

> **The trained weights are not published yet**
>
> There is no checkpoint to download, so reproducing the numbers below means training the
> model with the command above first. The weights will be released with the codebase.

## Reproducing the probe results

The linear-probe and retrieval numbers all come from one protocol. To compare against the published
tables, match it:

```bash
instanovo-fm evaluate \
    --checkpoint path/to/model_best.ckpt \
    evaluation.max_samples=200000 \
    evaluation.batch_size=256 \
    evaluation.random_state=42 \
    evaluation.embedding_pooling=[mean_pool] \
    evaluation.tasks_to_run=[linearprobetask,duplicateretrievaltask] \
    duplicateretrievaltask.max_samples=20000 \
    linearprobetask.use_project_split=false \
    linearprobetask.max_iter=5000
```

Two of those settings change what the numbers mean, not just their precision:

- **`use_project_split=false`** shares projects between the probe's train and test splits, so the
  scores measure what the embedding encodes rather than held-out generalisation. Each result
  records the regime it ran under, as `config.projects_shared_across_splits`.
- **The solver backend** is cuML on GPU, falling back to scikit-learn on CPU. Scores are not
  comparable between the two, so each result records which it used, as `config.backend`. Check
  that field before comparing runs.

## Figures

| Notebook | Panels |
|---|---|
| `notebooks/figure_1.ipynb` | Corpus composition — organisms, instruments, fragmentation, tiers |
| `notebooks/figure_3.ipynb` | Embedding UMAP, property zoom insets, probe barplots |
| `notebooks/figure_4.ipynb` | Integrated-gradients attribution of a masked fragment group |

Run them through the kernel `setup_kernel.sh` registers. The pins matter: `figure_3` ranks candidate
zoom regions numerically, and an unpinned numpy can order ties differently.

The attribution input for `figure_4` is committed under `data/`, because it cannot be regenerated
without the trained model.

## Downstream retrieval, rescue and run classification

The database-free identification and run-classification panels come from the evaluation tasks in
`src/instanovo_fm/eval/embed_eval_tasks/`, driven the same way as any other task:

```bash
instanovo-fm evaluate \
    --checkpoint path/to/model_best.ckpt \
    evaluation.tasks_to_run=[crosssetannotationtransfertask]
```

The datasets these tasks consume are not raw spectra — they carry anchor and query roles, so they
are built first, by the `scripts/create_*_dataset.py` builders. `scripts/discover_spectral_rescue_pairs.py`
selects the query/anchor pairs, and the two `plot_*_publication.py` scripts draw the panels.

Cross-set retrieval runs in two stages on purpose: the evaluator does the retrieval and writes
`cross_set_topk_candidates.csv`, then `scripts/compute_cross_set_evidence_metrics.py` scores the
evidence blocks offline from that file. The second stage is where the cost is, so it is resumable
and kept separate.

> **The evidence-metric blocks cannot be reproduced here yet**
>
> Those blocks score a query spectrum against the *theoretical* spectrum of its transferred
> peptide, and that scoring comes from `proteomics-mcp`, which is unpublished work by its author
> and is deliberately not included — see [Sanitisation](sanitisation.md#what-is-not-ported).
> Retrieval, rescue, and the observed-versus-observed metrics are unaffected and run without it.


## Reading the results

Evaluation writes one directory per task:

```
<output>/instanovo_fm/eval/embed_eval_results/<task>/task_summary.json
```

`task_summary.json` holds the headline metrics; `task_results.json` holds per-class detail.
