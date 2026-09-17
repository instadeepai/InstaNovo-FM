# InstaNovo-FM

**A self-supervised foundation model for proteomics tandem mass spectra**

<!-- Badges: update the Colab URL once the notebook is live -->
[![PyPI version](https://img.shields.io/pypi/v/instanovo-fm.svg)](https://pypi.org/project/instanovo-fm/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.09.03.747733-blue.svg)](https://doi.org/10.64898/2026.09.03.747733)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/instadeepai/InstaNovo-FM/blob/main/notebooks/getting_started_with_instanovo_fm.ipynb)

The official code repository for **InstaNovo-FM**, a self-supervised foundation model for
bottom-up proteomics. Unlike existing proteomics models that are trained for a single supervised
task (peptide identification, *de novo* sequencing, or fragment-intensity prediction),
InstaNovo-FM is an encoder-only transformer trained to **reconstruct masked regions of tandem mass
spectra without using any peptide-sequence labels**. The resulting frozen embeddings form a unified
representation space that transfers across datasets, instruments, and acquisition methods.

Publication: *Learning from tandem mass spectra at scale with a self-supervised foundation
model for proteomics*, bioRxiv, 3 September 2026.
[doi:10.64898/2026.09.03.747733](https://doi.org/10.64898/2026.09.03.747733)

<!-- TODO: add graphical abstract, e.g. docs/assets/graphical_abstract.png -->
<!-- ![Graphical Abstract](docs/assets/graphical_abstract.png) -->

## Highlights

- **Annotation-free pretraining.** Learns transferable spectral representations from raw MS/MS
  spectra with a physics-aware masked-reconstruction objective, without peptide sequence labels at any
  pretraining stage.
- **Trained at scale.** A diverse corpus of ~1.63 billion MS/MS spectra (1,625,276,573 scans)
  with 184.6 million high-confidence peptide-spectrum matches at 1% FDR, assembled from 92 public
  PRIDE submissions via an LLM-assisted metadata curation pipeline spanning 72 organisms and
  diverse instrumentation, fragmentation and digestion regimes.
- **One encoder, many tasks.** The same pretrained encoder drives database-free identification
  and spectrum rescue, PTM and glycan detection, and run-level classification straight from
  frozen embeddings, with no retraining.
- **Interpretable.** Attention and integrated-gradients analysis show the model recovers real
  fragmentation chemistry: ion-ladder complementarity, isotope and neutral-loss relationships, and
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

## System requirements

### Software dependencies

Every dependency is pinned in [`pyproject.toml`](pyproject.toml), locked in [`uv.lock`](uv.lock) and
listed with its licence in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The pins are the
versions the manuscript's results were produced with. The ones that most affect a reinstall:

| dependency | version |
| --- | --- |
| Python | 3.10–3.13 |
| `instanovo` | 1.2.2 (exact) |
| `torch` | ≥2.5,<2.9 on Linux and Windows; ≥2.5,<2.8 on macOS |
| `numpy` | 2.2.6 (exact) |
| `pyarrow` | 22.0.0 (exact) |
| `typer` | 0.27.2 (exact) |
| `polars` | ≥1.12.0 |
| `scikit-learn` | ≥1.5.2 |

### Operating systems

| platform | status |
| --- | --- |
| Ubuntu 24.04 LTS (GitHub Actions `ubuntu-latest`), Python 3.11 | tested on every push — lint, type-check and the full test suite, see [CI](.github/workflows/ci.yml) |
| Ubuntu 24.10, Python 3.11 | used for development and for the timings quoted below |
| macOS, Windows | the dependency pins carry platform-specific `torch` ranges so an install resolves, but neither is covered by CI and neither produced any published result |

### Hardware

No non-standard hardware is required to install the package, run the [demo](#demo), or regenerate
the paper figures — all of those run on an ordinary desktop CPU. A CUDA GPU is required in practice
for pretraining, *de novo* fine-tuning and large embedding or evaluation sweeps.

- **GPU.** Optional for inference on modest inputs, effectively required for training. The published
  model was trained on NVIDIA H100 nodes under 🤗 Accelerate — multi-GPU, fp16 mixed precision,
  batch size 1024 — per [`configs/foundational.yaml`](src/instanovo_fm/configs/foundational.yaml)
  and [`configs/accelerate/aichor.yaml`](src/instanovo_fm/configs/accelerate/aichor.yaml).
- **Linear probes.** These run on GPU through cuML and fall back to scikit-learn on CPU. The two
  backends do not give comparable scores, so every result records which it used as `config.backend`
  — see [GPU-accelerated linear probes](docs/gpu-probes.md).
- **Disk.** Checkpoints download on demand into `~/.cache/instanovo-fm/`: 342 MiB per foundation
  checkpoint, 659 MiB per *de novo* checkpoint. A default Linux install also pulls 4.1 GB of NVIDIA
  CUDA wheels transitively via `torch` (8.4 GB of site-packages in total); [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) has a
  CPU-only recipe that avoids them.
- **Reference machine.** The timings quoted in this README come from a laptop with a 20-core
  x86-64 CPU, 30 GB RAM and an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), running Ubuntu 24.10.
  Where a timing is CPU-only it says so, and was measured with the GPU hidden via
  `CUDA_VISIBLE_DEVICES=""`.

## Reproducing the figures

The three notebooks in `notebooks/` regenerate the paper figures from the committed
inputs in `data/`, no checkpoint or GPU needed.

```bash
./setup_kernel.sh   # uv sync --group figures, then registers a Jupyter kernel
```

Then open a notebook and select the **InstaNovo-FM (figures)** kernel.

## Installation

We support Python 3.10–3.13.

```bash
pip install instanovo-fm                # model, dataset pipeline and evaluation harness
pip install "instanovo-fm[interpret]"   # adds UMAP visualization
pip install "instanovo-fm[clustering]"  # adds the EvoC clustering eval task
```

**Typical install time:** about **10 minutes** on the reference machine — a cold `pip` cache on a
normal desktop over a home connection, measured at 10 min 11 s. Nearly all of that is download, not
compilation: no wheel is built from source, and the bulk is the ~4 GB of CUDA libraries `torch`
pulls in. A warm cache, or the CPU-only recipe in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), is substantially faster.

## Demo

[`sample_data/`](sample_data/) holds three small MGF files, committed so the software can be
demonstrated without downloading a dataset first: one spectrum from the manuscript's Fig. 4, a
270-spectrum reference library, and 20 query spectra. [`scripts/demo.py`](scripts/demo.py) runs the
three things the paper claims the frozen encoder is good for, each at a scale that fits on a laptop:

```bash
pip install instanovo-fm
python scripts/demo.py
```

It embeds a spectrum, sequences it *de novo*, and then identifies the 20 queries against the library
by embedding nearest neighbour alone — no peptide database involved.

### Expected output

```
Device: cuda

[1] Embedding fig4_TGVELGKPTHFTVNAK.mgf (TGVELGKPTHFTVNAK)
    Shape:          (1, 768)
    Norm:           1.000
    First 5 values: [-0.0084, 0.0005, -0.0135, -0.0212, -0.0062]

[2] De novo sequencing fig4_TGVELGKPTHFTVNAK.mgf
    Annotated:  TGVELGKPTHFTVNAK
    Predicted:  TGVELGKPTHFTVNAK   (match)
    Log prob:   -0.000  (uncalibrated score 1.000)

[3] Database-free retrieval: 20 queries vs 270-spectrum library
    Top-1 correct:       11/20 (55%)
    Mean cosine (hit):   0.992
    Mean cosine (miss):  0.988
    OK   query ATETVDNKDIER         -> ATETVDNKDIER         cos 0.992
    OK   query GSIAGVR              -> GSIAGVR              cos 0.992
    MISS query GVAMNPVDHPHGGGNHQHIGK -> AIVHHYEEC(+57.02)AERVK cos 0.989
```

The run is deterministic, and CPU and GPU agree to every digit printed — only the `Device:` line
differs. Three things in that output are worth reading carefully.

- **The embedding norm is 1.000 by construction.** Mean-pooled embeddings are L2-normalized so that
  a dot product between two of them is a cosine similarity. It is a normalization convention, not a
  measure of spectral quality or identification confidence.
- **A score of 1.000 is not a 100% chance of being right.** The *de novo* call is correct here, but
  the beam-search log probability is an uncalibrated model score. Turning scores like it into
  something you can threshold needs FDR control — the Winnow step in
  [the tutorial notebook](notebooks/getting_started_with_instanovo_fm.ipynb).
- **55% top-1 is the honest number, and the misses are the interesting part.** Every query's peptide
  *is* in the library, so a perfect search would score 20/20. Note how little separates a hit from a
  miss: 0.992 mean cosine against 0.988. Nearest-neighbour similarity alone is a candidate
  generator, not an identification — which is exactly why the paper's retrieval and rescue pipeline
  validates candidates with precursor mass, raw-spectrum similarity and fragment-ion evidence before
  transferring a label. This demo deliberately stops short of that step so the gap is visible.

### Expected run time

Measured on the reference laptop, with the checkpoints already cached:

| | cached run |
| --- | --- |
| GPU (RTX 4070 Laptop) | **15 seconds** |
| CPU only | **40 seconds** |

The **first** run additionally downloads the two checkpoints — 342 MiB encoder, 659 MiB *de novo* —
into `~/.cache/instanovo-fm/`. That part is bound by your connection rather than by the machine: a
cold-cache run on the reference laptop took 44 seconds in total, but on a slower link the download
alone can take a few minutes. Budget a couple of minutes the first time and seconds thereafter.

A GPU is not required at this size; it matters for batches of thousands of spectra, not for 291.

### Running it on your own data

`scripts/demo.py` takes a directory holding files named as `sample_data/` does. To run the model
over your own spectra instead, use the batch entry points under [Quick start](#quick-start) — they
accept any input `SpectrumDataFrame` reads, including MGF, mzML, mzXML, CSV and parquet — and see
[Getting started](docs/getting_started.md) for converting raw files and training on your own data.

## Quick start

Everything is driven by module entry points and Hydra configs from
`src/instanovo_fm/configs/`.

### Load a pretrained checkpoint

```python
from instanovo_fm.model.encoder import FoundationModel

FoundationModel.get_pretrained()
# ['instanovo-fm-v0.1.0', 'instanovo-fm-lcfm-ts-pa-v0.1.0', ...]

model, config = FoundationModel.from_pretrained("instanovo-fm-v0.1.0")
```

The ids differ only by training corpus, masking strategy and whether the pairwise
attention bias is on, so `describe_pretrained` says which is which:

```python
FoundationModel.describe_pretrained("instanovo-fm-v0.1.0")
# {'remote': '...', 'corpus': 'LCFM', 'masking': 'thompson_span with isotope co-masking',
#  'pairwise_bias': False, 'layers': 12, 'model_dimension': 768, 'parameters': '89.5M', ...}

FoundationModel.describe_pretrained()          # every checkpoint, keyed by id
```

The `corpus` field names a confidence tier of the pretraining corpus. They nest, from
everything that was collected down to only the most confidently identified spectra:

| tier | what it is |
|---|---|
| **ACFM** | *All Confidence* — every MS/MS scan in the corpus, ~1.63B, the vast majority with no peptide annotation at all. What the self-supervised objective can learn from. |
| **LCFM** | *Low Confidence* — the labelled subset: 184.6M PSMs at run-specific 1% FDR. "Low" means least-stringently filtered, not unreliable, and it is the broadest labelled tier. |
| **MCFM** | *Medium Confidence* — a nested subset of LCFM, ranked by a composite confidence score and thresholded. |
| **HCFM** | *High Confidence* — the strictest subset, nested inside MCFM. |

The released checkpoints are trained on LCFM, with one MCFM model for the corpus-scale
comparison. HCFM is used for evaluation rather than pretraining, and ACFM is not released.

| id | corpus | masking | PA bias | layers | params |
|---|---|---|---|---|---|
| `instanovo-fm-v0.1.0` | LCFM | Thompson-span | no | 12 | 89.5M |
| `instanovo-fm-lcfm-ts-pa-v0.1.0` | LCFM | Thompson-span | yes | 12 | 89.5M |
| `instanovo-fm-lcfm-sa-nopa-v0.1.0` | LCFM | signal-aware | no | 12 | 89.5M |
| `instanovo-fm-lcfm-sa-pa-v0.1.0` | LCFM | signal-aware | yes | 12 | 89.5M |
| `instanovo-fm-mcfm-90k-v0.1.0` | MCFM | Thompson-span | no | 9 | 40M |

The first row is the published model: every TS-noPA number in the paper comes from it.
The next three complete the masking/attention-bias factorial, and the last is the
corpus-scale comparison baseline.

The de novo sequencers — the foundation encoder plus an InstaNovo decoder — load the same
way, from `DownstreamDeNovo`:

```python
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo

DownstreamDeNovo.describe_pretrained()
model, config = DownstreamDeNovo.from_pretrained("instanovo-fm-denovo-v0.1.0")
```

| id | encoder | notes |
|---|---|---|
| `instanovo-fm-denovo-v0.1.0` | fine-tuned | the published sequencer, benchmarked against IN v1.2, Casanovo and XuanjiNovo |
| `instanovo-fm-denovo-frozen-v0.1.0` | frozen | retains ~85% of the fine-tuned peptide recall |
| `instanovo-fm-denovo-scratch-v0.1.0` | from scratch | the no-pretraining control |

All three run 2.5M steps at batch size 128, warming up over the first 100K steps to a
learning rate of 5e-5; the fine-tuned variant unfreezes the encoder at step 100K.

By id, the checkpoint is downloaded from this repository's
[Releases](https://github.com/instadeepai/InstaNovo-FM/releases) and cached under
`~/.cache/instanovo-fm/`. A path or a `.ckpt` filename loads from disk instead:

```python
model, config = FoundationModel.from_pretrained("checkpoints/model_best.ckpt")
```

The registry is [`src/instanovo_fm/models.json`](src/instanovo_fm/models.json). It covers
the published model, the four cells of the masking/attention-bias factorial, the MCFM
scaling baseline, and the three de novo sequencers — see
[Pretrained weights & data](#pretrained-weights--data) for the licence they carry.

### Extract embeddings and run the evaluation tasks

```bash
uv run python -m instanovo_fm.eval.embed_evaluation \
  --config-name foundational
```

The spectrum embedding is the mean of the final-layer hidden states over the non-padding
peak tokens, excluding the latent token. Downstream tasks live in
`src/instanovo_fm/eval/embed_eval_tasks/`: linear probes, duplicate retrieval, clustering,
attention and integrated-gradients attribution. Each is runnable the same way.

### Train

```bash
uv run python -m instanovo_fm.trainer.train \
  --config-name foundational
```

## Downstream applications

InstaNovo-FM's frozen embeddings are designed to be reused across tasks. Examples demonstrated in
the paper:

- **De novo peptide sequencing:** fine-tune or attach a decoder, competitive with supervised
  baselines on held-out biological datasets.
- **Database-free identification & rescue:** retrieve peptide identities for query spectra via
  embedding nearest-neighbours, including spectra unassigned by database search.
- **PTM & glycan analysis:** linear probes on frozen embeddings detect phosphorylation and
  glycosylation and resolve coarse glycan composition.
- **Run-level classification:** aggregate per-spectrum embeddings to classify technical and
  biological run conditions (e.g. digestion enzyme, treatment) without any peptide identifications.

> _TODO: add links to example notebooks / tutorials for each application._

## Pretrained weights & data

- **Pretraining corpus:** [`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo)
  on HuggingFace, under [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).
  Assembled from 92 public PRIDE submissions, with accessions in
  [`assets/table_s1_accessions.txt`](assets/table_s1_accessions.txt), and uniformly reprocessed
  with FragPipe (v22.0) / MSFragger (v4.1). The confidence tiers are described under
  [Load a pretrained checkpoint](#load-a-pretrained-checkpoint). Each of the three *labelled*
  tiers — LCFM, MCFM and HCFM — ships in two forms: `splits/` holds the
  quality-filtered, peptide-disjoint 80/10/10 partitions the model was trained and evaluated on,
  and `by_project/` holds the tier before filtering and splitting, one directory per accession,
  so alternative partitions can be derived. The central peptide registry of split assignments
  ships alongside, so the partitions can be reproduced and extended. ACFM itself is not released.
- **Model checkpoints:** attached to the
  [`v0.1.0` release](https://github.com/instadeepai/InstaNovo-FM/releases/tag/v0.1.0) under
  CC BY-NC-SA 4.0 (see [License](#license)). Eight in all — five foundation models and
  three de novo sequencers — registered in
  [`models.json`](src/instanovo_fm/models.json) and loaded by id with `from_pretrained`,
  which caches under `~/.cache/instanovo-fm/`. See
  [Load a pretrained checkpoint](#load-a-pretrained-checkpoint).
- **Embeddings:** [`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings)
  on HuggingFace, under CC BY-NC-SA 4.0 (the model's terms, since they are derived through it).
  Two configs of held-out LCFM test spectra, mean-pooled from `instanovo-fm-v0.1.0`: `100k`
  is the point set published as Figure 3 and carries its exact coordinates, `1M` is a larger
  draw carrying two 2-D and two 3-D UMAP layouts (not yet mentioned in
  [v2 of the preprint](https://www.biorxiv.org/content/10.64898/2026.09.03.747733v2)). Each
  row is a 768-d vector plus the metadata identifying its spectrum. The peak arrays are not
  duplicated — join to the corpus on `usi`, which is **not unique** here, so take first
  occurrences. An interactive explorer over the same embedding space is hosted at
  [instadeepai.github.io/InstaNovo-FM](https://instadeepai.github.io/InstaNovo-FM).

  ```python
  from datasets import load_dataset

  fig3 = load_dataset("InstaDeepAI/InstaNovo-FM-embeddings", "100k", split="train")
  ```

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
│   ├── release/          # HuggingFace upload, dataset card, audit scripts
│   └── demo.py           # embed, sequence and retrieve -- see Demo above
├── notebooks/            # figure-reproduction notebooks (figure_1, figure_3, figure_4)
├── data/                 # committed inputs for those notebooks
├── sample_data/          # three small MGFs, so the demo runs offline
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

An interactive explorer for the frozen embedding space, including a figure viewer and a UMAP
browser over ~100,000 held-out LCFM spectra, is hosted at
[instadeepai.github.io/InstaNovo-FM](https://instadeepai.github.io/InstaNovo-FM). The
embeddings behind it are published as
[`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings),
so the space can be analysed rather than only browsed.

Design and operational notes live in `docs/`. A hosted docs site is planned. _(TODO: add docs site
URL.)_ Until then, the guides live in [`docs/`](docs/):

**Tutorials:** start here

- [Getting started with the Foundation Model](docs/getting_started.md): install, train a small
  model on your own spectra, and read embeddings out of it.

**How-to guides:** task-oriented

- [Train the Foundation Model](docs/foundation_model_training.md): the full set of training and
  evaluation options, multi-device training, and experiment tracking.
- [Reproduce the Foundation Model results](docs/reproducing_paper_results.md): the published
  configuration and the evaluation protocol behind the paper's numbers.
- [GPU-accelerated linear probes](docs/gpu-probes.md): installing cuML, and why it is not a
  locked dependency.

**Explanation:** background

- [The InstaNovo Foundation Model](docs/foundation_model.md): what the model is, how masked-peak
  reconstruction works, and what the embeddings encode.
- [Downstream de novo sequencing](src/instanovo_fm/downstream/de_novo_sequencing/README.md): the
  FM encoder plus an InstaNovo decoder, and the staged unfreeze schedule.
- [Sanitisation of ported code](docs/sanitisation.md): what is removed from the internal
  repository on the way here, and how to review a port separately from its sanitisation.

**Reference**

- `instanovo-fm --help`, and `instanovo-fm train|evaluate|denovo --help` for per-command options.
- Configs live in [`src/instanovo_fm/configs/`](src/instanovo_fm/configs/), every setting is
  overridable with Hydra syntax on the command line.

**For developers**

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): full dependency licence list and a CPU-only
  install recipe.
- See [Development](#development) below for the test suite and pre-commit hooks.

## Development

```bash
uv sync --group dev        # pytest, ruff, mypy and pre-commit
uv run pytest

pre-commit install        # ruff, ruff-format, mypy, whitespace and key-leak hooks
pre-commit run --all-files
```

`ruff` and `mypy` in the `dev` group are pinned to the versions
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) uses, so a local run and
[CI](.github/workflows/ci.yml) use the same tool versions.

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a
pull request. _(TODO: add CONTRIBUTING.md and issue/PR templates.)_

## Citation

If you use InstaNovo-FM in your research, please cite:

```bibtex
@article{instanovofm,
  title   = {Learning from tandem mass spectra at scale with a self-supervised foundation
             model for proteomics},
  author  = {Nieuwoudt, Mechiel and Reverenna, Marco and Patel, Divanisha
             and Catzel, Rachel and Houngue, Isaac H.J. and Daniel, Jemma
             and Eloff, Kevin and Santos, Alberto and Lopez Carranza, Nicolas
             and Jenkins, Timothy P. and Van Goey, Jeroen
             and Kalogeropoulos, Konstantinos},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.64898/2026.09.03.747733},
  url     = {https://doi.org/10.64898/2026.09.03.747733},
  note    = {Preprint}
}
```

## License

| artifact | licence |
|---|---|
| **Code** in this repository | [Apache License 2.0](LICENSE.md) |
| **Model checkpoints** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution, non-commercial, share-alike |
| **Corpus-production code** ([Figshare](https://doi.org/10.6084/m9.figshare.33368752)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Dataset** ([`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo)) | [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/), the spectra derive from public PRIDE submissions |
| **Embeddings** ([`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings)) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): derived from those spectra through a non-commercially-licensed model, so the model's terms carry over |


The Apache-2.0 grant covers this repository's source alone. It does **not** extend to the
third-party binary packages an install fetches, some of which are proprietary. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Dependencies are pinned to the versions used for the manuscript results to support reproducibility,
rather than automatically receiving later patch releases. A vulnerability
scan will therefore report real findings; [`SECURITY.md`](SECURITY.md) enumerates them with a
reachability assessment, and [`vex.openvex.json`](vex.openvex.json) publishes the same assessment in
machine-readable [OpenVEX](https://openvex.dev) form. **If you are deploying this code rather than
reproducing the paper with it, do not use these pins.**

## Acknowledgements

Developed by:

- [InstaDeep](https://www.instadeep.com/)
- [Novo Nordisk Foundation Biotechnology Research Institute for the Green Transition](https://www.dtu.dk/), Technical University of Denmark
- [Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine), Technical University of Denmark
- [Center for Translational Protein Design](https://www.dtu.dk/)
- [Delft University of Technology](https://www.tudelft.nl/en/) & the [Kavli Institute of Nanoscience](https://kavli.tudelft.nl/)

Built on public proteomics data from the [PRIDE](https://www.ebi.ac.uk/pride/) repository and the
broader open-proteomics community.
