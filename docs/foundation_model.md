# The InstaNovo Foundation Model

InstaNovo-FM is a **self-supervised, encoder-only transformer** that learns representations of
MS/MS spectra by masking peaks and reconstructing them. It does not decode peptides: there is no
decoder, no beam search and no diffusion anywhere in `src/instanovo_fm/`. Its output is a
*spectrum embedding*, which downstream probes and retrieval tasks then evaluate.

This is what distinguishes it from the de novo models in the rest of this repository. InstaNovo
and InstaNovo+ map a spectrum to a sequence and are trained on annotated data. InstaNovo-FM maps a
spectrum to a vector and needs no annotations at all — peptide labels are used only at evaluation
time, to ask what the embedding turned out to encode.

## Why encoder-only

Masked reconstruction gives a training signal from any spectrum, annotated or not. That matters
because the annotated fraction of public proteomics data is small and biased toward
easy-to-identify spectra: tryptic, unmodified, well-fragmented. Training on the raw corpus instead
lets the model see the rest of it.

The cost is that nothing forces the representation to be *about* peptides. Whether it is remains an
empirical question, which is why evaluation here is a battery of probes rather than a single
accuracy number.

## Data flow

```
Raw MS/MS
  → SpectrumDataFrame              (shared with the rest of InstaNovo)
  → FoundationalDataProcessor      src/instanovo_fm/data/data.py
  → masking                        src/instanovo_fm/data/masking.py
  → batching
  → encoder                        src/instanovo_fm/model/encoder.py
  → prediction heads               src/instanovo_fm/model/heads.py
  → loss                           src/instanovo_fm/trainer/losses.py
  → embeddings
  → evaluation tasks               src/instanovo_fm/eval/embed_eval_tasks/
```

Each stage owns one concern, and the boundaries are contracts rather than conventions. In
particular: masking hides peaks from the encoder and gives ground truth only to the loss, so the
encoder never sees a masked value; the encoder itself knows nothing about labels or peptides, and
the prediction heads sit on top without altering its state; and evaluation loads a checkpoint and
computes embeddings without touching weights, masking or training state.

## Masking

Masking is the training signal, so its design is the main modelling decision in the project.

**Thompson-span (`thompson_span`, "TS")** is the deployed strategy. It samples contiguous spans of
3–4 peaks using a Beta(0.5, 0.5) prior tempered by intensity, targeting ~26 % of peaks. Spans
rather than isolated peaks matter because masking a single peak from a fragment ladder is close to
free to reconstruct by interpolation — the neighbouring ladder members give it away.

Two refinements ship with the deployed model:

- **Isotope co-masking** (`masking.include_isotopes = True`). A monoisotopic peak and its isotope
  envelope are masked together. Masking only the monoisotopic peak leaves the +1 isotope in place
  at a known offset of `k/z` Da, which reveals the answer.
- **Mass blurring** (`masking.blur_sigma_da = 10.0`). The reconstruction target is a Gaussian over
  m/z rather than a delta, so the loss does not demand more precision than the instrument provides.

An alternative, `signal_aware_fragment`, masks whole theoretical fragment groups. It is stronger on
some reconstruction metrics but weaker on the frozen-probe battery — see the factorial comparison
below.

## Positional encoding

The deployed model uses **no positional encoding** (`positional_encoding.type: 'none'`). A spectrum
is a set of (m/z, intensity) pairs, not a sequence: position in a sorted peak list carries no
physical meaning, and m/z itself is already encoded by the peak encoder. Rotary, ALiBi, relative
and sinusoidal variants are all implemented under `model/positional/` and were ablated; none
improved on omitting them.

Peaks are embedded by a **multiscale Fourier encoder** (192 frequencies over periods 0.001–1.5 in
normalised m/z, plus 1024 RBFs), which is what carries m/z information into the model.

## The deployed model

The published model is **LCFM TS·noPA** — trained on the LCFM corpus with Thompson-span masking and
**no pairwise attention bias**.

| | |
|---|---|
| Layers | 12 |
| Model dimension | 768 |
| Attention heads | 12 |
| Feed-forward dimension | 3072 |
| Dropout | 0.1 |
| Parameters | ~89.5 M |
| Positional encoding | none |
| Masking | `thompson_span`, spans 3–4, 26 %, isotope co-masking, σ = 10 Da |
| Training | 200 001 steps (6 epochs) |

Note that `src/instanovo_fm/configs/model/foundation_base.yaml` is the **ablation baseline** (9 layers,
feed-forward 1024), not the deployed model. The published configuration is that baseline plus the
overrides `model.n_layers=12 model.dim_feedforward=3072 dataset=lcfm`, exactly as recorded in
`manifests/lcfm_ts34_nopa.yaml`.

### Why TS·noPA

A 2 × 2 factorial over masking strategy (signal-aware vs Thompson-span) and pairwise attention bias
(present vs absent) was run on LCFM. TS·noPA wins the frozen-probe battery and, more strikingly,
develops far more structurally specialised attention heads:

| Variant | Fragment-type macro F1 | Structural heads / 12 |
|---|---|---|
| SA·noPA | 0.793 | 1 |
| SA·PA | 0.804 | 1 |
| **TS·noPA** | **0.865** | **8** |
| TS·PA | 0.757 | 4 |

Adding the pairwise bias *hurts* under Thompson-span masking. The interpretation we favour is that
an explicit m/z-difference bias supplies the fragment-ladder relationship the model would otherwise
have to learn, and having it handed over means the attention heads never specialise.

## Evaluation

Because there is no accuracy metric for a representation, the model is judged by what can be
recovered from frozen embeddings:

- **Linear probes** — fragmentation method, instrument, precursor charge, PTM presence, plus
  regressions on m/z, peptide mass and hydrophobicity. A linear probe is used deliberately: it
  measures what the embedding makes *linearly* available, not what a sufficiently large head could
  extract.
- **Duplicate retrieval** — given a spectrum, retrieve other spectra of the same peptide. This
  tests whether the embedding is organised by peptide identity rather than by acquisition
  conditions.
- **Peak-type classification** and **cross-spectrum ion identity** — whether peak-level
  representations encode ion series, and whether the *same* ion in two different spectra lands in
  the same place.
- **Integrated-gradients attribution** — which input peaks the model actually used to reconstruct a
  masked group, and whether those peaks correspond to chemically meaningful relationships.

## The published embeddings

The frozen embeddings of the held-out LCFM test split are published as
[`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings):
a `100k` config carrying the Figure 3 point set with its exact published coordinates, and a
`1M` config carrying two 2-D and two 3-D UMAP layouts (not yet mentioned in
[v2 of the preprint](https://www.biorxiv.org/content/10.64898/2026.09.03.747733v2)).
Each row is the 768-d mean-pooled vector plus the metadata identifying its spectrum.

Use them to analyse the embedding space without re-running inference. To reproduce them
instead, `scripts/release/prepare_embeddings.py` turns the eval harness's `embeddings.h5`
into the published shards.

## Corpora

- **MCFM** — a curated, high-confidence subset.
- **LCFM** — the full corpus, ~230 k steps of training data, split **peptide-disjoint** 80/10/10 so
  no peptide appears in more than one split.

The two behave differently and the difference is informative rather than a defect: LCFM produces
stronger *linear decodability* on nearly every probe, while MCFM produces stronger *retrieval*.
Scale buys decodability; curation buys a cleaner neighbourhood structure.
