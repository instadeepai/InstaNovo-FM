---
license: cc-by-nc-sa-4.0
task_categories:
  - feature-extraction
tags:
  - proteomics
  - mass-spectrometry
  - embeddings
  - foundation-model
pretty_name: InstaNovo-FM spectrum embeddings
size_categories:
  - 1M<n<10M
configs:
  - config_name: 100k
    data_files: 100k/*.parquet
  - config_name: 1M
    data_files: 1M/*.parquet
---

# InstaNovo-FM spectrum embeddings

Frozen embeddings of held-out tandem mass spectra, produced by
[InstaNovo-FM](https://github.com/instadeepai/InstaNovo-FM) — a self-supervised foundation
model trained to reconstruct masked regions of MS/MS spectra without peptide-sequence
labels.

Each row is one spectrum: a 768-dimensional mean-pooled embedding, the search-engine and
acquisition metadata that identifies it, and the UMAP coordinates used in the paper.

Publication: *Learning from tandem mass spectra at scale with a self-supervised foundation
model for proteomics*, bioRxiv, 3 September 2026 (v2).
[doi:10.64898/2026.09.03.747733](https://doi.org/10.64898/2026.09.03.747733)

## The two configs

```python
from datasets import load_dataset

fig3 = load_dataset("InstaDeepAI/InstaNovo-FM-embeddings", "100k", split="train")
big  = load_dataset("InstaDeepAI/InstaNovo-FM-embeddings", "1M", split="train")
```

| config | spectra | what it is |
|---|---|---|
| `100k` | 100,000 | the point set published as **Figure 3**. Carries `figure3_umap_x` / `figure3_umap_y`, the exact coordinates in the paper |
| `1M` | 1,000,000 | a larger draw from the same held-out split. Carries two 2-D and two 3-D UMAP layouts, not yet mentioned in [v2 of the preprint](https://www.biorxiv.org/content/10.64898/2026.09.03.747733v2) |

Both are the LCFM **test** split — held out from training — embedded with the released
checkpoint `instanovo-fm-v0.1.0` using `mean_pool`.

**The two configs are not disjoint, and not redundant.** The 100k spectra are also in the
1M set. They come from two separate extraction runs, and their embeddings were checked
against each other: over 2,000 spectra sampled from the overlap, every vector is
**bit-identical** (maximum absolute difference 0.0, minimum cosine similarity 1.0). Use
`100k` to reproduce the figure, `1M` for anything that wants more data.

## Columns

- **`embedding`** — 768 × float32, mean-pooled over the spectrum's peak representations.
- **52 metadata fields** — `usi`, `sequence`, `precursor_mz`, `precursor_charge`,
  `retention_time`, `hyperscore`, `expectation`, `search_project`, `experiment_name`,
  `protein`, `header` (the instrument filter line), and so on. Several arrive from the
  eval harness as strings even when numeric; cast them rather than assuming a dtype.
- **UMAP coordinates** — `figure3_umap_x/y` in `100k`; `native_x/y`, `native3_x/y/z`,
  `transform_x/y`, `transform3_x/y/z` in `1M`.
- **Five derived columns** — `sequence_length`, `n_peaks`, `peak_center_of_mass`,
  `peak_spread`, `top_duplicate_peptides`. These are computed after extraction, so they
  are not in the model's own output; they were recomputed here and checked against the
  published Figure 3 table over all 100,000 rows:

  | column | agreement with the published values |
  |---|---|
  | `sequence_length` | 100,000 / 100,000 exact |
  | `n_peaks` | 100,000 / 100,000 exact |
  | `peak_center_of_mass` | bit-identical, maximum difference 0.0 |
  | `peak_spread` | bit-identical, maximum difference 0.0 |
  | `top_duplicate_peptides` | 99,728 / 100,000 — see below |

  `n_peaks` counts the model's 200 peak slots, not the full spectrum: the stored peak
  list runs to 800 peaks (median 235), so counting that would give a plausible number
  meaning something else.

  `top_duplicate_peptides` is the rank among the ten most repeated peptides, or 10 for
  everything else. It is population-dependent by definition, so a spectrum ranks
  differently in the two configs. The 272 rows that differ from the published column are
  all ties at equal counts — ranks 0-6 match exactly, ranks 7 and 8 are both count 70 and
  swap, and rank 9 is a tie at count 66 so one peptide falls either side of the cut. Ties
  are broken alphabetically here, which is deterministic; the original order was not
  recorded.

The `1M` layouts differ in a way worth knowing before using them:

| columns | layout |
|---|---|
| `transform_*` | the published Figure 3 layout extended to a million points. Figure 3's own spectra keep their exact published coordinates; the rest were placed into that same space with `umap-learn`'s `.transform()` |
| `native_*` | an independent `cuml.manifold.UMAP` fit over all 1,000,000 embeddings. A *different* embedding, not a denser Figure 3 — Procrustes r = 0.858 against the transform layout globally, but about 11% 20-NN neighbourhood overlap |

A 3-component UMAP is a separate optimisation, not an extra axis on a 2-D fit, so the
`*3_x/y/z` triples are not the 2-D columns plus a third value. Use each triple together.

### Seven columns the Figure 3 table has and this does not, yet

`annotation_ratio`, `backbone_coverage`, `median_ppm_error`, `n_fragment_groups_metric`,
`signal_intensity_ratio`, `match_metrics` and `spectrum_quality` all require
theoretical-spectrum generation, peak matching and quality scoring. They are absent from
this revision rather than approximated: the annotation path needs the harness's own
modified-sequence handling, which 31% of these rows require, and a reimplementation would
risk plausible-but-wrong values for a third of the data. They are planned for a later
revision, generated by the harness with the parameters this extraction recorded
(20 ppm tolerance, a/b/y ions, H2O/NH3/SO3/H3PO4 losses, isotopes to 4).

## What is not here

- **The spectra themselves.** Peak *m/z* and intensity arrays are excluded; the corpus is
  published properly at
  [`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo), joinable
  on `usi`.
- **Per-peak tensors** (`peak_mask`, `spectra_mask`, `targets`, `precursors`) — model
  plumbing, reconstructible from the corpus.
- **Unlabelled spectra.** These are PSM-annotated rows only.

## `usi` is not a unique key

1,000,000 rows carry 999,720 distinct `usi` values. The repeats are timsTOF spectra: their
identifier is really the *(frame, scan)* pair, and the USI keeps only `scan`, so two
spectra from different frames collapse onto one identifier. In the `100k` config, 100,000
rows carry 99,997 distinct values.

Join on `usi` with first-occurrence semantics, or deduplicate first. Nothing here is
ordered by `usi`, and `sample_idx` is a position within one extraction run — it is not
comparable between the two configs.

## Reproducing

The embeddings come from the released checkpoint, so they are reproducible from the model
and the corpus:

```python
from instanovo_fm.model.encoder import FoundationModel
model, config = FoundationModel.from_pretrained("instanovo-fm-v0.1.0")
```

The UMAP layouts, the figure and an interactive explorer over the same spectra are
described in the [repository](https://github.com/instadeepai/InstaNovo-FM), which also
holds the script that produced these shards
(`scripts/release/prepare_embeddings.py`).

## Licence

**CC BY-NC-SA 4.0.** These are derived from public PRIDE submissions through a
non-commercially-licensed model, so the model's terms carry over. The underlying spectra
remain subject to [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).

## Citation

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
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.03.747733v2},
  note    = {Preprint}
}
```
