---
license: other
license_name: embl-ebi-terms-of-use
license_link: https://www.ebi.ac.uk/about/terms-of-use/
task_categories:
  - feature-extraction
tags:
  - proteomics
  - mass-spectrometry
  - peptide-sequencing
pretty_name: InstaNovo-FM training corpus
configs:
  # Fully explicit names: no bare tier name exists, so a config can never be
  # ambiguous about which representation it refers to.
  #
  # splits/ uses data_dir because the filenames carry train/validation/test and
  # datasets detects them. by_project/ must NOT: with data_dir, split detection
  # runs inside the directory, finds no split keyword in the run names, and
  # collapses everything into a split called "train" -- advertising a training
  # split that does not exist. data_files with `split: full` names it honestly.
  - config_name: hcfm_splits
    default: true
    data_dir: splits/hcfm
  - config_name: mcfm_splits
    data_dir: splits/mcfm
  - config_name: lcfm_splits
    data_dir: splits/lcfm
  - config_name: hcfm_by_project
    data_files:
      - split: full
        path: by_project/hcfm/**/*.parquet
  - config_name: mcfm_by_project
    data_files:
      - split: full
        path: by_project/mcfm/**/*.parquet
  - config_name: lcfm_by_project
    data_files:
      - split: full
        path: by_project/lcfm/**/*.parquet
  - config_name: peptide_registry
    data_files:
      - split: full
        path: peptide_registry.parquet
---

# InstaNovo-FM training corpus

Tandem mass spectra with peptide-spectrum-match labels, uniformly reprocessed from
public PRIDE submissions, used to pretrain and evaluate InstaNovo-FM.

## Layout

Spectra are grouped into three **confidence tiers**. Each is a Foundational Model
dataset named for how stringently its peptide-spectrum matches (PSMs) were filtered:
**LCFM** (Low), **MCFM** (Medium) and **HCFM** (High). The names are relative, not
absolute — every labelled tier consists of high-confidence PSMs, and "low confidence"
marks LCFM as the broadest, least stringently filtered one, from which the stricter
subsets are derived. They are nested: HCFM ⊂ MCFM ⊂ LCFM. A fourth tier, **ACFM** (All
Confidence), is the unlabelled superset and is not published here.

Each tier is published twice, once under `splits/` and once under `by_project/`.
Which you want depends on whether you are consuming the corpus or rebuilding it:

- **`splits/`** — the train/validation/test partitions the model actually consumed:
  quality-filtered, shuffled, and peptide-disjoint. Take this to reproduce or extend
  the published results.
- **`by_project/`** — the same tier *before* filtering and splitting, one directory per
  PRIDE accession. Take this to apply your own quality criteria or derive your own
  partitions, which `splits/` cannot support because the filtering is lossy.

The tier sits *inside* the folder rather than above it — `splits/lcfm/`, not
`lcfm/splits/`. That nesting is deliberate: it means a download pattern like `--include
"by_project/*"` cannot stray outside the folder you named. Nested the other way,
`lcfm/*` would have matched both folders at once and quietly handed you ~900 GB
containing two overlapping copies of the same spectra, one filtered and one not.

```
InstaDeepAI/InstaNovo
│
├── splits/                                                   FILTERED · SHUFFLED · PEPTIDE-DISJOINT
│   ├── hcfm/                                                 11 files ·  11.4 GB ·   3,670,113 rows
│   │   ├── hcfm-train-00000-of-00007.parquet … 00006-of-00007
│   │   ├── hcfm-validation-00000-of-00001.parquet
│   │   └── hcfm-test-00000-of-00003.parquet … 00002-of-00003
│   ├── mcfm/                                                 44 files ·  52.5 GB ·  18,255,265 rows
│   │   ├── mcfm-train-00000-of-00029.parquet … 00028-of-00029
│   │   ├── mcfm-validation-00000-of-00001.parquet
│   │   └── mcfm-test-00000-of-00014.parquet … 00013-of-00014
│   └── lcfm/                                                 454 files · 467.1 GB · 181,777,591 rows
│       ├── lcfm-train-00000-of-00293.parquet … 00292-of-00293
│       ├── lcfm-validation-00000-of-00019.parquet … 00018-of-00019
│       └── lcfm-test-00000-of-00142.parquet … 00141-of-00142
│
├── by_project/                                               COMPLETE TIER · NOT FILTERED · NOT SPLIT
│   ├── hcfm/                                                 15,166 files ·  11.4 GB ·   3,684,448 rows
│   │   ├── PXD000561/                                        82 accessions in every tier
│   │   │   ├── Adult_Adrenalgland_Gel_Elite_49_f01.parquet
│   │   │   └── …                                             one file per instrument run
│   │   └── PXD000865/ …
│   ├── mcfm/                                                 15,244 files ·  51.7 GB ·  18,422,236 rows
│   │   └── PXD000561/ …                                      same runs, fewer rows each
│   └── lcfm/                                                 15,286 files · ~450 GB · 184,607,213 rows
│       └── PXD000561/ …
│
├── peptide_registry.parquet                                  5,613,657 peptides and their split
│
└── manifests/
    └── empty_runs.csv                                        162 runs with no PSMs at their threshold
```

**Pick one tier, from one folder.** Because the tiers are nested, a second tier
re-downloads the same spectra at a stricter threshold — and `splits/*` or
`by_project/*` fetches all three, roughly 531 GB and 510 GB.

### `splits/` — start here

`train`, `validation` and `test` parquet, exactly as the model consumed them. The
partitions are **peptide-disjoint** 80/10/10: a peptide sequence appears in only one
of the three, so evaluation does not reward memorisation. Rows are shuffled and have
passed the quality filters below.

Rows per split, measured:

| | train | validation | test |
|---|---:|---:|---:|
| `hcfm_splits` | 2,459,391 (67.0%) | 178,997 (4.9%) | 1,031,725 (28.1%) |
| `mcfm_splits` | 11,703,040 (64.1%) | 790,417 (4.3%) | 5,761,808 (31.6%) |
| `lcfm_splits` | 117,230,014 (64.5%) | 7,716,481 (4.3%) | 56,831,096 (31.3%) |

The 80/10/10 ratio is over **peptides**, not spectra: the registry assigns each peptide
to one split and every spectrum of that peptide follows it. Peptides differ in how many
spectra they have, and test peptides carry disproportionately many — 11.5% of peptides
but ~31% of rows — so the test partition is about 2.7x its nominal share and validation
about half of its own.

Use this to reproduce or extend the published results.

### `by_project/` — the input the splits came from

Each tier before filtering and splitting. Published because the filtering is lossy:
rows dropped by the quality gates are not recoverable from `splits/`.

Use this to apply different quality criteria, or to re-derive the partitions with
`peptide_registry.parquet` (see below).

### Quality filters applied to `splits/` but not to `by_project/`

| field | kept |
|---|---|
| retention time | <= 10800 s |
| lower isolation offset | <= 300 Da |
| precursor charge | 0-7 **inclusive** (0 is kept) |
| precursor *m/z* | <= 2000 |
| modification annotation | only if resolvable to a UNIMOD identifier |

A null in any numeric field **passes** its condition, so a reimplementation that
discards nulls instead would produce a different dataset. These tiers contain no nulls
in any filtered field, so the two behave identically here — but the distinction matters
if you apply the same criteria to new data.

Measured effect on LCFM (184,607,213 -> 181,777,591 rows, 98.5% kept): retention time
removes ~2.54 M, the unresolved-modification condition ~385 k, lower offset 8,147, and
the charge and *m/z* bounds remove **nothing** — no row in the corpus exceeds either.

The unresolved annotations are modifications the pipeline could not resolve to a UNIMOD identifier (`[IN:<digits>]`; 175 tokens over ids 3000–3174, predominantly N-glycans on asparagine but **not** exclusively — `K[IN:3174]` is on lysine). Glycopeptides whose glycan *does* have a
UNIMOD identifier are retained: 100 distinct UNIMOD modifications occur across 30.7% of
the corpus.

### `peptide_registry.parquet`

The split assignment of every peptide in the corpus. It is what makes the
peptide-disjoint partitions reproducible, and it lets new data be partitioned
consistently with these splits rather than at random — apply it to `by_project/`,
or to your own spectra, to keep a held-out set genuinely held out.

### Joining `by_project/` to the registry

Join on **`registry_key`** — not on `sequence`, and not on `unmodified_peptide`:

```python
from datasets import load_dataset

# hcfm is the smallest tier (11.4 GB); swap in mcfm or lcfm when you need more.
runs = load_dataset("InstaDeepAI/InstaNovo", "hcfm_by_project", split="full")
registry = load_dataset("InstaDeepAI/InstaNovo", "peptide_registry", split="full")

labelled = runs.to_polars().join(
    registry.to_polars(), left_on="registry_key", right_on="peptide", how="left"
)
```

Both calls download and cache from the Hub, so nothing needs fetching by hand.
`to_polars()` materialises the tier in memory, and `mz_array` and `intensity_array`
dominate that, so call `runs.select_columns(["registry_key", "experiment_name",
"scan"])` first if you only need the split assignment. To work through one accession at
a time rather than a whole tier, pass `streaming=True` and filter on `experiment_name`,
or fetch single files directly:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "InstaDeepAI/InstaNovo",
    "by_project/hcfm/PXD000561/Adult_Adrenalgland_Gel_Elite_49_f01.parquet",
    repo_type="dataset",
)
```

`registry_key` is `unmodified_peptide` with every `I` rewritten to `L`, which is the
convention the registry itself uses: its `peptide` column contains no `I` at all.
Isoleucine and leucine are isobaric, so a peptide pair differing only by I/L is
indistinguishable to MS/MS, and collapsing them into one key is what stops one variant
training while the other tests.

Mind the direction. Collapsing `L` to `I` instead yields keys that match nothing, and a
failed join returns *no rows* rather than visibly wrong ones — which reads naturally as
"these peptides are new".

The collapse applies to the key only. `sequence` and `unmodified_peptide` keep their
original I and L residues, and the model was trained on those unnormalised sequences.

There is no `normalised_peptide` column. Earlier internal copies carried one that was
never populated; it is removed here so nothing invites a join that would silently match
nothing.

## Confidence tiers

The tiers are nested subsets at increasing PSM-confidence thresholds: LCFM is the
full labelled corpus, MCFM and HCFM are progressively stricter. MCFM and HCFM inherit
LCFM's split assignments, so a peptide has the same split in every tier.

The unlabelled **ACFM** tier is not published here. It is approximately 5.7 TB and
comprises every MS/MS scan from the same raw files, so it is reconstructible from the
PRIDE accessions with the conversion pipeline deposited at Figshare, and is otherwise
available from the authors on request.

## Loading a specific tier

The configs above name every tier and flavour, so nothing is inferred from the
directory layout:

```python
from datasets import load_dataset

load_dataset("InstaDeepAI/InstaNovo", "hcfm_splits")         # train/validation/test
load_dataset("InstaDeepAI/InstaNovo", "lcfm_splits", split="test")
load_dataset("InstaDeepAI/InstaNovo", "hcfm_by_project")     # one "full" split
```

To fetch files without loading them, select by path — tier and flavour are
directory prefixes:

```bash
hf download InstaDeepAI/InstaNovo --repo-type dataset --include "hcfm/*"
```

## File naming

Split files follow the Hub's sharding convention,
`{split}-{index:05d}-of-{total:05d}.parquet`:

```
splits/hcfm/hcfm-train-00000-of-00007.parquet
splits/hcfm/hcfm-validation-00000-of-00001.parquet
```

The `-of-{total}` suffix makes an incomplete download self-evident, and the
zero-padding sorts correctly in a plain lexicographic listing.

`by_project/` deliberately does **not** use that convention. It has no train,
validation or test split, and any of those keywords in a filename would make the
Hub advertise a split that does not exist. Its files are named
`data-{index:05d}-of-{total:05d}.parquet` within each project accession directory,
so per-project selection stays possible:

```
by_project/lcfm/PXD012345/<instrument-run-name>.parquet
```

## Versioning

`v0.1` is the first release, and the one the paper's results were produced from. Pin it
rather than tracking `main`, so a later addition to the corpus cannot change what you
fetch:

```python
from datasets import load_dataset
ds = load_dataset("InstaDeepAI/InstaNovo", "hcfm_splits", revision="v0.1")
```

`main` moves as tiers are extended or corrected; a tag does not.

## Licence

The spectra derive from public submissions to
[PRIDE](https://www.ebi.ac.uk/pride/), so use of this dataset is governed by the
[EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).

The code that produced the corpus is available on
[Figshare](https://doi.org/10.6084/m9.figshare.33368752) under a CC BY 4.0 licence, and the
model checkpoints are CC BY-NC-SA 4.0.

## Citation

Nieuwoudt, M., Reverenna, M., Patel, D., Catzel, R., Houngue, I. H. J., Daniel, J., Eloff, K.,
Santos, A., Lopez Carranza, N., Jenkins, T. P., Van Goey, J., & Kalogeropoulos, K. (2026).
*Learning from tandem mass spectra at scale with a self-supervised foundation model for
proteomics*. bioRxiv. <https://doi.org/10.64898/2026.09.03.747733>

```bibtex
@article{instanovofm,
  title   = {Learning from tandem mass spectra at scale with a self-supervised foundation model for proteomics},
  author  = {Nieuwoudt, Mechiel and Reverenna, Marco and Patel, Divanisha and Catzel, Rachel and Houngue, Isaac H.J. and Daniel, Jemma and Eloff, Kevin and Santos, Alberto and Lopez Carranza, Nicolas and Jenkins, Timothy P. and Van Goey, Jeroen and Kalogeropoulos, Konstantinos},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.64898/2026.09.03.747733},
  url     = {https://doi.org/10.64898/2026.09.03.747733},
  note    = {Preprint}
}
```

Code, including the pipeline that produced these files, is at
<https://github.com/instadeepai/InstaNovo-FM>.
