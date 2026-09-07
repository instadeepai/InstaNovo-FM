# InstaNovo-FM

**A self-supervised foundation model for proteomics tandem mass spectra**

<!-- Badges — update the PyPI and Colab URLs once those are live -->
[![PyPI version](https://img.shields.io/badge/pypi-coming--soon-lightgrey.svg)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.09.03.747733-blue.svg)](https://doi.org/10.64898/2026.09.03.747733)
[![Open In Colab](https://img.shields.io/badge/Colab-coming--soon-lightgrey.svg)](#)

> ⚠️ **Starting-point README.** This is an initial scaffold. Sections marked _TODO_ (installation commands, model weights, download links, benchmarks, DOI, documentation site) need to be filled in as the code, checkpoints, and data are released.

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
- **Trained at scale.** A diverse corpus of ~1.80 billion MS/MS spectra with 184.6 million
  high-confidence peptide-spectrum matches, assembled via an LLM-assisted metadata curation
  pipeline spanning 68 organisms, 16 instrument models, and multiple fragmentation and digestion
  regimes.
- **Frozen embeddings that transfer.** A single frozen encoder supports *de novo* peptide
  sequencing, database-free identification and spectrum rescue, PTM and glycan detection, and
  run-level classification.
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

The pins live in `pyproject.toml` under the `figures` dependency group and are
enforced, not advisory: each notebook checks its environment in the first code cell
and refuses to run on a different numpy. This is
not fussiness — `figure_3` ranks candidate zoom windows whose separation scores
frequently tie, and numpy 1.26 and 2.0 broke those ties differently, silently
selecting different regions for the Enzyme, PTM, Organism, Chemical-labelling and
Detector panels. The ranking is now a total order, so the selection is reproducible,
and the check catches the rest.

The notebooks do not install anything themselves. Installing from a cell cannot fix
a version mismatch anyway: once numpy is imported, `pip install numpy==…` does not
change the module already loaded in the kernel.

## Installation

> _TODO: replace with real instructions once the package is published._

```bash
# Recommended: create an isolated environment (Python 3.10+)
# e.g. with uv, conda, or venv

# From PyPI (planned)
# pip install instanovo-fm

# From source
git clone https://github.com/<org>/InstaNovo-FM.git
cd InstaNovo-FM
pip install -e .
```

### Installing this accepts NVIDIA's proprietary licence, not only Apache-2.0

This project is Apache-2.0, but **a default install on Linux fetches closed-source
NVIDIA binaries, and installing them means accepting NVIDIA's licence terms.** Nothing
here asks for a GPU, and neither `torch` nor any `nvidia-*` package appears in
`pyproject.toml` — they arrive transitively:

| declared here | pulls | which pulls |
|---|---|---|
| `instanovo==1.2.2` | `accelerate>=1.6.0` | `torch` |
| `torch>=2.5,<2.9` (transitive) | — | **14 `nvidia-*` CUDA wheels**, 12 of them declaring an NVIDIA proprietary licence |
| `gpu` extra: `cuml-cu12` | `cupy-cuda12x`, `nvidia-nvcomp-cu12` | more CUDA runtime; needs NVIDIA's package index |

The proprietary libraries include **cuBLAS**, **cuDNN**, **cuSPARSELt**, **cuSOLVER** and
**cuFFT**. cuDNN and cuBLAS are the kernel backends
`torch.nn.functional.scaled_dot_product_attention` dispatches to, so on NVIDIA hardware
they execute inside the model's forward pass. Each wheel ships NVIDIA's terms as
`License.txt` in its `.dist-info` directory.

This matters most if you **redistribute** a container image or a built environment: you
are then redistributing NVIDIA's binaries under NVIDIA's terms, which are more
restrictive than Apache-2.0. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
the full package list and a CPU-only recipe that avoids all of them. The dataset
pipeline, the figure notebooks and the release scripts need no GPU.

## Quick start

> _TODO: confirm the public API and CLI. The snippets below are illustrative placeholders._

### Embed spectra (Python)

```python
from instanovo_fm import InstaNovoFM  # TODO: confirm import path

model = InstaNovoFM.from_pretrained("instanovo-fm")  # TODO: confirm checkpoint name
model.eval()

# spectra: your MS/MS data (e.g. mzML/MGF loaded into the expected format)
embeddings = model.embed(spectra)  # frozen, fixed-dimensional spectrum embeddings
```

### Command-line interface

```bash
# TODO: confirm CLI once implemented
# instanovo-fm embed --input spectra.mzML --output embeddings.parquet
```

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

> _TODO: add download links and licensing terms once released._

- **Model checkpoints** — _pending._
- **Embeddings** — _pending._
- **Pretraining corpus** — assembled from public PRIDE submissions and uniformly reprocessed with
  FragPipe. Confidence tiers: **ACFM** (~1.8B spectra, unlabelled), **LCFM** (184.6M PSMs at 1% FDR),
  **MCFM** and **HCFM** (progressively stricter subsets). Peptide-disjoint 80/10/10 splits.
  _TODO: add accession list / repository link._

## Repository structure

> _TODO: update to match the actual layout as code lands._

```
InstaNovo-FM/
├── instanovo_fm/     # model, tokenization, training, and inference code
├── notebooks/        # tutorials and worked examples
├── docs/             # documentation and assets
├── tests/            # unit and integration tests
├── sample_data/      # small example spectra for a quick start
├── pyproject.toml
├── LICENSE
└── README.md
```

## Documentation

A hosted docs site is planned. _(TODO: add docs site URL.)_ Until then, the guides live in
[`docs/`](docs/):

**Tutorials** — start here

- [Getting started with the Foundation Model](docs/getting_started.md) — install, train a small
  model on your own spectra, and read embeddings out of it.

**How-to guides** — task-oriented

- [Train the Foundation Model](docs/foundation_model_training.md) — the full set of training and
  evaluation options, multi-device training, and experiment tracking.
- [Reproduce the Foundation Model results](docs/reproducing_paper_results.md) — the published
  configuration and the evaluation protocol behind the paper's numbers.
- [GPU-accelerated linear probes](docs/gpu-probes.md) — installing cuML, and why it is not a
  locked dependency.

**Explanation** — background

- [The InstaNovo Foundation Model](docs/foundation_model.md) — what the model is, how masked-peak
  reconstruction works, and what the embeddings encode.
- [Downstream de novo sequencing](src/instanovo_fm/downstream/de_novo_sequencing/README.md) — the
  FM encoder plus an InstaNovo decoder, and the staged unfreeze schedule.
- [Sanitisation of ported code](docs/sanitisation.md) — what is removed from the internal
  repository on the way here, and how to review a port separately from its sanitisation.

**Reference**

- `instanovo-fm --help`, and `instanovo-fm train|evaluate|denovo --help` for per-command options.
- Configs live in [`src/instanovo_fm/configs/`](src/instanovo_fm/configs/); every setting is
  overridable with Hydra syntax on the command line.

**For developers**

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — full dependency licence list and a CPU-only
  install recipe.
- See [Development](#development) below for the test suite and pre-commit hooks.

## Development

> _TODO: confirm tooling (uv/poetry, pre-commit, pytest)._

```bash
# Set up a dev environment and run the test suite
# pip install -e ".[dev]"
# pytest
```

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

> _TODO: confirm license terms._ Following the InstaNovo project, the intended split is code under
> the **Apache License 2.0** and model checkpoints under a **Creative Commons
> Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license. Update this section once
> finalized.

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
