# Getting started with the Foundation Model

InstaNovo-FM turns an MS/MS spectrum into a vector. It is trained by hiding peaks and
reconstructing them, so it needs **no peptide annotations** — any collection of spectra will do.
This tutorial trains a small model on your own data and then reads embeddings out of it.

If you want to know what the model is before running it, read
[The InstaNovo Foundation Model](foundation_model.md) first.

> **This tutorial trains its own small model**
>
> Training from scratch is what shows you the training signal, so that is what the steps below
> do — the result is not the published model. If you only want embeddings, load the published
> weights instead and skip to the embedding section:
>
> ```python
> from instanovo_fm.model.encoder import FoundationModel
>
> model, config = FoundationModel.from_pretrained("instanovo-fm-v0.1.0")
> ```
>
> To retrain the published model yourself, see
> [Reproduce the Foundation Model results](reproducing_paper_results.md).

## Installation

Install the package, as in the [Installation section of the README](../README.md#installation):

```bash
pip install instanovo-fm
```

Then check the CLI is present:

```bash
instanovo-fm --help
```

You should see three commands: `train`, `evaluate` and `denovo`. If you would rather not
install the console script, `python -m instanovo_fm.cli` works identically.

> **A GPU matters more here than for inference**
>
> Masked-peak reconstruction trains over hundreds of peaks per spectrum for many thousands of
> steps, so CPU-only training is only practical for the small demonstration below. The linear
> probes also run on GPU through cuML and fall back to scikit-learn on CPU, which gives
> different scores — every result records which backend it used, as `config.backend`. See
> [GPU-accelerated linear probes](gpu-probes.md).
>
> Note also that a default install on Linux pulls closed-source NVIDIA CUDA wheels
> transitively via `torch`; see the README for a CPU-only recipe that avoids them.

## Preparing spectra

Training reads a `SpectrumDataFrame`. Convert your `.mgf`, `.mzml` or `.mzxml` files once, using
the `instanovo` CLI that ships as a dependency:

```bash
mkdir -p ./my_spectra
instanovo convert './raw/*.mgf' ./my_spectra --name my_spectra --partition train
```

`target` is a *folder*, not a file — conversion writes sharded parquet into it. `--name` and
`--partition` are both required. Repeat with `--partition valid` for your validation spectra.

Peptide labels are ignored during training — they are only used later, if you want to probe what
the embeddings encode.

## Training a small model

Point the config at your files and shrink the model so it trains in minutes rather than days:

```bash
instanovo-fm train \
    dataset.train_path='./my_spectra/*train*.parquet' \
    dataset.valid_path='./my_spectra/*valid*.parquet' \
    model.n_layers=2 \
    model.dim_model=128 \
    model.n_heads=4 \
    train_batch_size=16 \
    predict_batch_size=16 \
    training_steps=200 \
    checkpoint_interval=100 \
    post_training_evaluation.enabled=False
```

Watch the reconstruction loss fall. That is the whole training signal: the model is predicting the
*m/z* of peaks it was not allowed to see.

Those last four overrides exist only because this is a small run, and each one fails in a way that
is hard to read if you leave it out:

- **`train_batch_size`** defaults to 1024 and the training loader drops incomplete batches, so a
  dataset smaller than 1024 spectra produces no batches at all and training stops with a bare
  `StopIteration`. Set it below your spectrum count.
- **`predict_batch_size`** governs validation and wants the same treatment.
- **`checkpoint_interval`** defaults to 10,000 steps, so a 200-step run writes no checkpoint and
  the evaluation step below has nothing to load.
- **`post_training_evaluation.enabled`** is on by default and runs the full task battery against
  `dataset.test_path` when training finishes. This tutorial never sets a test split, so leaving it
  on ends the run with a `FileNotFoundError`. Set a `dataset.test_path` instead if you do want it.

> **This is a demonstration, not the published model**
>
> The published model is 12 layers at d=768, trained for 230,000 steps. See
> [Reproduce the Foundation Model results](reproducing_paper_results.md) for its exact
> configuration.

## Reading embeddings out

Evaluation loads a checkpoint, computes embeddings and probes what they encode:

```bash
instanovo-fm evaluate \
    --checkpoint ./checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.tasks_to_run=[embeddingstatisticstask]
```

`embeddingstatisticstask` needs no labels — it reports the geometry of the embedding space
(effective rank, anisotropy, mean pairwise similarity). Results land in:

```
<output>/instanovo_fm/eval/embed_eval_results/<task>/task_summary.json
```

If your spectra *are* annotated, the probe tasks become available and will tell you what is
linearly decodable from the frozen embeddings:

```bash
instanovo-fm evaluate \
    --checkpoint ./checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.tasks_to_run=[linearprobetask]
```

## Where to go next

- [Training the Foundation Model](foundation_model_training.md) — the full set of training and
  evaluation options.
- [Reproduce the Foundation Model results](reproducing_paper_results.md) — the published
  configuration and the evaluation protocol its numbers come from.
- [GPU-accelerated linear probes](gpu-probes.md) — installing cuML, and why it is not a locked
  dependency.
