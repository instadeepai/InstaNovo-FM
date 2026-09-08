# How-to: Train the Foundation Model

The Foundation Model learns spectrum representations by masking peaks and reconstructing them, so
training needs no peptide annotations. Any collection of MS/MS spectra readable as a
`SpectrumDataFrame` will do.

For what the model is, see [The InstaNovo Foundation Model](foundation_model.md).

## Training

```bash
instanovo-fm train
```

This uses `src/instanovo_fm/configs/foundational.yaml`, which composes the architecture from
`configs/model/foundation_base.yaml` and the corpus from `configs/dataset/`.

Override any setting with Hydra syntax as trailing arguments:

```bash
instanovo-fm train \
    model=foundation_base \
    dataset=lcfm \
    model.n_layers=12 \
    model.dim_feedforward=3072
```

Those particular overrides are the published model — see
[Reproduce the Foundation Model results](reproducing_paper_results.md).

| Config | Role |
|---|---|
| `configs/foundational.yaml` | Training entry point |
| `configs/foundational_local.yaml` | Local variant, and the default for evaluation |
| `configs/model/foundation_base.yaml` | Architecture and masking |
| `configs/dataset/{lcfm,mcfm,hcfm}.yaml` | Corpora |

## Running across several devices

Training uses 🤗 Accelerate, so it scales across devices without code changes. Generate an
Accelerate config for the visible devices with `accelerate config` before launching.

> **Keep `dispatch_batches=True`**
>
>
> The training dataset is an **iterable** `SpectrumDataFrame`. With `dispatch_batches=False`
> every rank iterates it independently, and because filtering can leave rank shards with
> different lengths, one rank exits the loop early while the others wait forever at the next
> NCCL collective. Training hangs rather than failing, so it is easy to misread as a slow step.


## Evaluating a checkpoint

```bash
instanovo-fm evaluate --checkpoint path/to/model_best.ckpt --split test
```

`--checkpoint` and `--split` are shorthand for the `evaluation.checkpoint_path` and
`evaluation.split` overrides. Further Hydra overrides can be appended:

```bash
instanovo-fm evaluate \
    --checkpoint checkpoints/instanovo-foundational-base/model_best.ckpt \
    --split test \
    evaluation.batch_size=256 \
    evaluation.tasks_to_run=[linearprobetask,duplicateretrievaltask]
```

`evaluation.tasks_to_run` selects the battery; the available tasks live in
`src/instanovo_fm/eval/embed_eval_tasks/`.

This is the same code path as the module form used by the run scripts:

```bash
uv run python -m instanovo_fm.eval.embed_evaluation \
    --config-name foundational evaluation.checkpoint_path=...
```

Both call `run_evaluation()` in `src/instanovo_fm/eval/embed_evaluation.py`.

> **Probe scores depend on the protocol**
>
>
> Sample counts, solver iterations and the compute backend all affect probe results. To compare
> against the published tables, use
> [the documented protocol](reproducing_paper_results.md#reproducing-the-probe-results).

## What gets written where

| Path | Default | Lifetime |
|---|---|---|
| Checkpoints | `checkpoints/instanovo-foundational-base` | written into, never cleared |
| Evaluation results |  `instanovo_fm/eval/embed_eval_results` | one directory per task, overwritten per run |
| Embedding cache | `<output>/embeddings_<split>/` | only when `save_embeddings=True`; **reused on the next run** unless `force_regenerate_embeddings=True` |
| MLflow local fallback | `./mlruns` | only when the remote is unreachable and `mlflow_allow_local_fallback=True` |

The embedding cache is the one worth understanding. It is keyed by split, so a multi-split run
caches `embeddings_train/`, `embeddings_valid/` and `embeddings_test/` separately. It is also
reused silently: if you change anything that affects the embeddings themselves — the checkpoint,
the pooling, the sample budget — either set `force_regenerate_embeddings=True` or evaluate into a
fresh `output_dir`.

## Experiment tracking

Training logs to MLflow, and it is on by default. Configure it in the training config:

```yaml
mlflow_enabled: True                  # set False to disable all tracking
mlflow_tracking_uri:                  # leave blank for a local sqlite:///mlflow.db
mlflow_experiment_name:               # defaults to the current git branch
mlflow_workspace: instanovo           # must be lowercase
mlflow_allow_local_fallback: True     # fall back to ./mlruns if the remote rejects auth
mlflow_log_system_metrics: True       # GPU/CPU/memory utilisation
mlflow_log_datasets: True             # dataset name, split sizes and checksums
```

With no `mlflow_tracking_uri`, runs land in a local SQLite database and are viewable with
`mlflow ui`. For a remote server, set the URI and supply credentials through the standard MLflow environment
variables.

`mlflow_allow_local_fallback` is worth knowing about: when it is on and the remote server refuses
authentication, training continues and logs locally instead of failing. That keeps a long run
alive, but it also means a run you expect to find on the server may only exist on the training
machine, so check where a run actually landed before going looking for it.

Metrics are also written to the run's output directory as JSON, which does not depend on a tracking
server being reachable at all:

```
<output>/instanovo_fm/eval/embed_eval_results/<task>/task_summary.json
```
