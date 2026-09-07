# Getting started with the Foundation Model

InstaNovo-FM turns an MS/MS spectrum into a vector. It is trained by hiding peaks and
reconstructing them, so it needs **no peptide annotations** — any collection of spectra will do.
This tutorial trains a small model on your own data and then reads embeddings out of it.

If you want to know what the model is before running it, read
[The InstaNovo Foundation Model](foundation_model.md) first.

> **There is no pretrained checkpoint to download yet**
>
> The weights behind the paper are not published, so this tutorial trains a small model from
> scratch. That is enough to see the training signal and inspect the embedding space, but it is
> not the published model — see
> [Reproduce the Foundation Model results](reproducing_paper_results.md) for that configuration.

## Installation

Install from source, following the [Installation section of the README](../README.md#installation):

```bash
git clone https://github.com/instadeepai/InstaNovo-FM.git
cd InstaNovo-FM
uv sync
```

Then check the CLI is present:

```bash
instanovo-fm --help
```

You should see two commands, `train` and `evaluate`. If you would rather not install the console
script, `python -m instanovo_fm.cli` works identically.

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
    training_steps=200
```

Watch the reconstruction loss fall. That is the whole training signal: the model is predicting the
*m/z* of peaks it was not allowed to see.

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
<output>/instanovo/foundational/eval/embed_eval_results/<variant>/<task>/task_summary.json
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
