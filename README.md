# InstaNovo-FM

**A self-supervised foundation model for proteomics tandem mass spectra**

<!-- Badges — update the PyPI and Colab URLs once those are live -->
[![PyPI version](https://img.shields.io/badge/pypi-coming--soon-lightgrey.svg)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.09.03.747733-blue.svg)](https://doi.org/10.64898/2026.09.03.747733)
[![Open In Colab](https://img.shields.io/badge/Colab-coming--soon-lightgrey.svg)](#)

The official code repository for **InstaNovo-FM**, a self-supervised foundation model for
bottom-up proteomics. Unlike existing proteomics models that are trained for a single supervised
task (peptide identification, *de novo* sequencing, or fragment-intensity prediction),
InstaNovo-FM is an encoder-only transformer trained to **reconstruct masked regions of tandem mass
spectra without using any peptide-sequence labels**. The resulting frozen embeddings form a unified
representation space that transfers across datasets, instruments, and acquisition methods.

Publication: *Learning from tandem mass spectra at scale with a self-supervised foundation
model for proteomics* — bioRxiv, 3 September 2026.
[doi:10.64898/2026.09.03.747733](https://doi.org/10.64898/2026.09.03.747733)

<!-- TODO: add graphical abstract, e.g. docs/assets/graphical_abstract.png -->
<!-- ![Graphical Abstract](docs/assets/graphical_abstract.png) -->

## Highlights

- **Annotation-free pretraining.** Learns transferable spectral representations from raw MS/MS
  spectra with a physics-aware masked-reconstruction objective — no peptide sequence labels at any
  pretraining stage.
- **Trained at scale.** A diverse corpus of ~1.63 billion MS/MS spectra (1,625,276,573 scans)
  with 184.6 million high-confidence peptide-spectrum matches at 1% FDR, assembled from 92 public
  PRIDE submissions via an LLM-assisted metadata curation pipeline spanning 72 organisms and
  diverse instrumentation, fragmentation and digestion regimes.
- **One encoder, many tasks.** The same pretrained encoder drives database-free identification
  and spectrum rescue, PTM and glycan detection, and run-level classification straight from
  frozen embeddings, with no retraining. *De novo* sequencing is the one benchmark that
  fine-tunes it, and even frozen it retains 85% of the fine-tuned peptide recall.
- **Interpretable.** Attention and integrated-gradients analysis show the model recovers real
  fragmentation chemistry — ion-ladder complementarity, isotope and neutral-loss relationships, and
  chemically defined off-database ions.

## Model at a glance

| Property | Value |
| --- | --- |
| Architecture | Encoder-only transformer |
| Model dimension | 768 |
| Layers / heads | 12 / 12 (head dim 64) |
| Feedforward dimension | 3072 |
| Parameters | ~89.5M |
| Peak encoding | Multi-scale sinusoidal ($m/z$) + MLP (intensity) |
| Masking | Thompson-span masking with isotope co-masking (~30% fragment-group budget) |
| Reconstruction objective | Hierarchical classification over a 0.2 Da $m/z$ grid (group + offset) |
| Spectrum embedding | Mean-pooled final-layer peak-token hidden states |

## Reproducing the figures

The three notebooks in `notebooks/` regenerate the paper figures from the committed
inputs in `data/` — no checkpoint or GPU needed.

```bash
./setup_kernel.sh   # uv sync --group figures, then registers a Jupyter kernel
```

Then open a notebook and select the **InstaNovo-FM (figures)** kernel.

## Installation

We support Python 3.10–3.13 and use  [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/instadeepai/InstaNovo-FM.git
cd InstaNovo-FM

uv sync                    # model, dataset pipeline and evaluation harness
uv sync --group figures    # when you want to reproduce the figures
uv sync --extra interpret  # for UMAP visualization
```

## Quick start

Everything is driven by module entry points and Hydra configs from
`src/instanovo_fm/configs/`. 

### Extract embeddings and run the evaluation tasks

```bash
uv run python -m instanovo_fm.eval.embed_evaluation \
  --config-name foundational_local
```

The spectrum embedding is the mean of the final-layer hidden states over the non-padding
peak tokens, excluding the latent token. Downstream tasks live in
`src/instanovo_fm/eval/embed_eval_tasks/` — linear probes, duplicate retrieval, clustering,
attention and integrated-gradients attribution — and each is runnable the same way.

### Train

```bash
uv run python -m instanovo_fm.trainer.train \
  --config-name foundational_local
```

> _TODO: wire up the `train` / `evaluate` CLI._ `src/instanovo_fm/cli.py` builds a typer app,
> but nothing invokes it: there is no `[project.scripts]` entry, no `__main__.py`, and no
> `if __name__ == "__main__"` guard, so both `instanovo-fm …` and `python -m instanovo_fm.cli …`
> exit silently without running the requested command. The `eval/` modules above each carry
> their own `__main__` guard and do work.

## Downstream applications

InstaNovo-FM's frozen embeddings are designed to be reused across tasks. Examples demonstrated in
the paper:

- **De novo peptide sequencing** — fine-tune or attach a decoder; competitive with supervised
  baselines on held-out biological datasets.
- **Database-free identification & rescue** — retrieve peptide identities for query spectra via
  embedding nearest-neighbours, including spectra unassigned by database search.
- **PTM & glycan analysis** — linear probes on frozen embeddings detect phosphorylation and
  glycosylation and resolve coarse glycan composition.
- **Run-level classification** — aggregate per-spectrum embeddings to classify technical and
  biological run conditions (e.g. digestion enzyme, treatment) without any peptide identifications.

> _TODO: add links to example notebooks / tutorials for each application._

## Pretrained weights & data

- **Pretraining corpus** — [`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo)
  on HuggingFace, under [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).
  Assembled from 92 public PRIDE submissions — accessions in
  [`assets/table_s1_accessions.txt`](assets/table_s1_accessions.txt) — and uniformly reprocessed
  with FragPipe (v22.0) / MSFragger (v4.1). Confidence tiers: **ACFM** (~1.63B spectra,
  unlabelled), **LCFM** (184.6M PSMs at 1% FDR), **MCFM** and **HCFM** (progressively stricter
  subsets). Each of the three *labelled* tiers ships in two forms: `splits/` holds the
  quality-filtered, peptide-disjoint 80/10/10 partitions the model was trained and evaluated on,
  and `by_project/` holds the tier before filtering and splitting, one directory per accession,
  so alternative partitions can be derived. The central peptide registry of split assignments
  ships alongside, so the partitions can be reproduced and extended. ACFM itself is not released.
- **Model checkpoints** — will be published from this repository's
  [Releases](https://github.com/instadeepai/InstaNovo-FM/releases) under CC BY-NC-SA 4.0
  (see [License](#license)). _Not yet available._
- **Embeddings** — not released as files. An interactive explorer for the frozen embedding space
  is hosted at [instadeepai.github.io/InstaNovo-FM](https://instadeepai.github.io/InstaNovo-FM)
  (goes live with the repository).

## Repository structure

```
InstaNovo-FM/
├── src/instanovo_fm/
│   ├── configs/          # Hydra configs (model, dataset, evaluation, accelerate)
│   ├── data/             # data processing, masking, metadata and analysers
│   ├── model/            # peak encoder, transformer encoder, prediction heads
│   ├── trainer/          # training loop, losses, checkpointing
│   ├── eval/             # evaluation harness and embed_eval_tasks/
│   └── utils/
├── scripts/
│   ├── preprocessing/    # raw-file conversion, modification labelling, parquet IO
│   ├── splitting/        # tier construction, peptide-disjoint splits, shuffling
│   ├── verification/     # metadata, mass and normalisation checks
│   └── release/          # HuggingFace upload, dataset card, audit scripts
├── notebooks/            # figure-reproduction notebooks (figure_1, figure_3, figure_4)
├── data/                 # committed inputs for those notebooks
├── assets/               # accession list, modification dictionaries, residue masses
├── config/               # plotting configuration
├── docker/               # baseline-model images (Casanovo, XuanjiNovo)
├── docs/                 # design and operational notes
├── tests/                # unit and integration tests
├── pyproject.toml
├── uv.lock
├── setup_kernel.sh
├── CITATION.cff
├── LICENSE.md
└── README.md
```

## Documentation

An interactive explorer for the frozen embedding space — a figure viewer plus a UMAP browser
over ~100,000 held-out LCFM spectra — is hosted at
[instadeepai.github.io/InstaNovo-FM](https://instadeepai.github.io/InstaNovo-FM). It becomes
reachable when the repository goes public.

Design and operational notes live in `docs/`. A full documentation site is still planned
_(TODO: add docs site URL.)_ — tutorials (installation, first embedding, evaluation), how-to
guides (custom datasets, training, downstream tasks), a CLI and API reference, and explanations
of the architecture and benchmarks.

## Development

```bash
uv sync --group dev        # pytest, ruff and mypy
uv run pytest

pre-commit install        # ruff, ruff-format, mypy, whitespace and key-leak hooks
pre-commit run --all-files
```

`ruff` and `mypy` in the `dev` group are pinned to the versions
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) uses, so a local run and
[CI](.github/workflows/ci.yml) cannot disagree about what passes. `pre-commit` itself is not
a project dependency — install it separately (`uv tool install pre-commit` or `pipx install
pre-commit`).

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a
pull request. _(TODO: add CONTRIBUTING.md and issue/PR templates.)_

## Citation

If you use InstaNovo-FM in your research, please cite:

```bibtex
@article{instanovofm,
  title   = {Learning from tandem mass spectra at scale with a self-supervised foundation model for proteomics},
  author  = {Nieuwoudt, Mechiel and Reverenna, Marco and Patel, Divanisha and Catzel, Rachel and Houngue, Isaac H.J. and Daniel, Jemma and Eloff, Kevin and Santos, Alberto and Lopez Carranza, Nicolas and Jenkins, Timothy P. and Van Goey, Jeroen and Kalogeropoulos, Konstantinos},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.64898/2026.09.03.747733},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.03.747733v1},
  note    = {Preprint}
}
```

[`CITATION.cff`](CITATION.cff) carries the same metadata in machine-readable form, so
GitHub's "Cite this repository" button and reference managers pick it up directly.

## License

Following the InstaNovo project, the artifacts carry different terms:

| artifact | licence |
|---|---|
| **Code** in this repository | [Apache License 2.0](LICENSE.md) |
| **Model checkpoints** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — attribution, non-commercial, share-alike |
| **Corpus-production code** ([Figshare](https://doi.org/10.6084/m9.figshare.33368752)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Dataset** ([`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo)) | [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/) — the spectra derive from public PRIDE submissions |

The terms differ per artifact, so a use permitted for one is not necessarily permitted for
all of them.

The Apache-2.0 grant covers this repository's source alone. It does **not** extend to the
third-party binary packages an install fetches, some of which are proprietary — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the NVIDIA note above.

Model checkpoints are not published yet; the licence above is the term they will carry.

Dependency versions are pinned to the ones the manuscript's results were produced with, not to the
latest patched versions, because reproducing the paper is this repository's purpose. A vulnerability
scan will therefore report real findings; [`SECURITY.md`](SECURITY.md) enumerates them with a
reachability assessment, and [`vex.openvex.json`](vex.openvex.json) publishes the same assessment in
machine-readable [OpenVEX](https://openvex.dev) form. **If you are deploying this code rather than
reproducing the paper with it, do not use these pins.**

Apache-2.0 covers the code in this repository. It does **not** cover the third-party binaries an
install fetches. Twelve NVIDIA CUDA wheels arrive transitively and are governed by NVIDIA's
proprietary licence, so installing or redistributing this project means accepting those terms in
addition to Apache-2.0 — see [Installation](#installing-this-accepts-nvidias-proprietary-licence-not-only-apache-20)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Acknowledgements

Developed by:

- [InstaDeep](https://www.instadeep.com/)
- [Novo Nordisk Foundation Biotechnology Research Institute for the Green Transition](https://www.dtu.dk/), Technical University of Denmark
- [Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine), Technical University of Denmark
- [Center for Translational Protein Design](https://www.dtu.dk/)
- Delft University of Technology & the Kavli Institute of Nanoscience

Built on public proteomics data from the [PRIDE](https://www.ebi.ac.uk/pride/) repository and the
broader open-proteomics community.
