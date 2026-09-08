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

> **You can skip the training run**
>
> The published weights are a release asset, so reproducing the numbers below does not require
> training the model first. `FoundationModel.from_pretrained("instanovo-fm-v0.1.0")` fetches
> them, and `FoundationModel.describe_pretrained()` lists the ablation checkpoints alongside it.

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

The remaining published panels are produced by scripts that are not part of this repository.


## Reading the results

Evaluation writes one directory per task:

```
<output>/instanovo_fm/eval/embed_eval_results/<task>/task_summary.json
```

`task_summary.json` holds the headline metrics; `task_results.json` holds per-class detail.
