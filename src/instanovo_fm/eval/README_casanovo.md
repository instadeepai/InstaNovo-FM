# Casanovo de novo benchmark

Runs **Casanovo v5.0.0** and scores it with the *same* scorer as
InstaNovo and XuanjiNovo, so peptide recall is directly comparable across models.

Unlike the [upstream XuanjiNovo benchmark](README_xuanjinovo.md) — which needs its own Docker image
and three steps across two environments — Casanovo installs cleanly alongside InstaNovo, so this
runs **in-process, in one step**, in the normal `instanovo` environment.

```
parquet ──> load_spectrum_dataframe ──> filter_unpredictable_rows
                                              │
                                        preprocessing
                                              │
                                        beam_search_decode
                                              │
                            canonical prediction CSV ──> score_predictions ──> summary.json
```

| File | Role |
|------|------|
| `predict_casanovo_de_novo.py` | Casanovo-specific: model loading, preprocessing, batched beam-search decode |
| `_predict_de_novo_common.py` | Model-agnostic: the canonical CSV schema, notation remapping, upfront filtering, dual-denominator scoring |
| `run_baseline_de_novo.py` | Config-driven multi-dataset runner + aggregated results row |

The split is the point. Everything that decides a *number* — which rows are eligible, how residues
are canonicalised, which denominator is reported — lives in `_predict_de_novo_common.py` and is
shared with every other baseline. `predict_casanovo_de_novo.py` only knows how to make Casanovo emit
a peptide.

---

## 1. Running it

Single dataset:

```bash
uv run python -m instanovo_fm.eval.predict_casanovo_de_novo \
  --parquet_path "$VALIDATION_ROOT/biological/annotated/dataset-immuno-*.parquet" \
  --output_path de_novo_results/casanovo/immuno.csv \
  --checkpoint_path checkpoints/casanovo_v5_0_0.ckpt \
  --n_beams 5 --score
```

All datasets in an inference config (the normal path):

```bash
uv run python -m instanovo_fm.eval.run_baseline_de_novo \
  -cn casanovo run_name=casanovo_v5_biological num_beams=5
```

`configs/inference/casanovo.yaml` inherits `inference/denovo.yaml` — the same 8 biological +
9 ninespecies-v1 dataset list the InstaNovo and downstream de novo runs use — and sets
`model: casanovo`, `num_beams: 5`, `batch_size: 64`. One job sequences and scores every dataset with
a single loaded model, writes a per-dataset CSV, and appends one wide aggregated row.

The checkpoint resolves through `resolve_checkpoint`: an `s3://` URI is downloaded to
`checkpoints/` and cached, an existing local path is used as-is, otherwise `--checkpoint_url` is
downloaded (default: the Casanovo GitHub release). `MODELS` in `run_baseline_de_novo.py` holds the
canonical checkpoint + URL pair, so the config need not carry either.

Casanovo itself is an optional dependency — `_require_casanovo()` raises with an install hint
(`uv pip install casanovo`) rather than an opaque `ImportError` at some later line.

---

## 2. Casanovo's own preprocessing, sourced from Casanovo

Unlike the [embedding benchmark](README_baseline_embeddings.md), which pushes every model through the
shared `FoundationalDataProcessor` so encoders are compared on identical inputs, de novo sequencing
gives each model its **published** pipeline. The question here is "how well does Casanovo sequence
this data", not "how well does its encoder do on our inputs".

That pipeline is not reimplemented. `_casanovo_preprocessing` instantiates Casanovo's own
`DeNovoDataModule` and takes two things off it:

```python
data_module = DeNovoDataModule(lance_dir="", max_charge=max_charge)
return data_module.preprocessing_fn, data_module.valid_charge
```

`preprocessing_fn` is the ordered `spectrum -> spectrum` chain; `valid_charge` is the set of accepted
precursor charges. Sourcing them from upstream means a Casanovo version bump changes our
preprocessing automatically, and there is no chance of a reimplementation drifting from the real
thing. `--min_intensity` is the one override (default: Casanovo's 0.01).

Each row becomes a `spectrum_utils.MsmsSpectrum` and runs the chain. It returns `None` — the
spectrum is **unpredictable** — when the charge is outside `valid_charge`, or when the chain raises
`ValueError` (Casanovo's low-quality and intensity filters discard the spectrum).

`max_charge` comes from the checkpoint's `hparams`. A config value is passed through
`reconcile_max_charge`, which takes the smaller of the two and warns when a larger config value is
clamped down — mirroring the InstaNovo predictor, so no baseline is accidentally given a wider charge
range than it was trained for.

---

## 3. Decoding

`predict_dataframe` walks the frame accumulating preprocessed spectra into a buffer, and flushes each
full batch through `_decode_batch`. Only **predictable** spectra enter the buffer, so the batch is
never padded with rows that would be discarded anyway.

`_decode_batch` builds the three tensors Casanovo's `beam_search_decode` expects — right-padded `mzs`
and `intensities` to the batch's longest spectrum, and a `(B, 3)` precursor tensor:

```python
precursors[j] = [(prec_mz - PROTON) * charge, charge, prec_mz]
```

`PROTON` is imported from `casanovo.data.db_utils` rather than redefined, so the neutral-mass
derivation matches Casanovo's exactly. Only the **top beam** is kept; its peptide score becomes
`log_probs = log(max(score, 1e-10))`, which is what the confidence-curve metrics (AUC,
recall@5% FDR) rank on.

The peptide is stored in **Casanovo's own notation** and translated at scoring time, not at decode
time. Predictions on disk stay faithful to what the model emitted; canonicalisation is the scorer's
job and is applied uniformly across models.

Beam width is `--n_beams` (default: whatever the checkpoint carries, usually 1). The benchmark uses 5.

---

## 4. What makes the score comparable

### The canonical CSV schema

Every baseline writes `PREDICTION_CSV_COLUMNS`:

```
prediction_id, predictions, targets, log_probs, predictable, precursor_mz, precursor_charge, group
```

`predictable` is the load-bearing column — it is how a dropped spectrum stays visible instead of
disappearing from the denominator. `group` resolves to `frag_type`, else `experiment_name`, else
`all`, which is what enables per-group metric breakdowns.

### Upfront filtering

`filter_unpredictable_rows` drops rows the model *structurally* cannot handle, before decoding,
mirroring the InstaNovo predictor:

- precursor charge outside `[1, max_charge]`
- target peptides containing residues the model cannot emit

The second is decided by `build_model_vocab`, which pushes Casanovo's residue vocabulary through the
remapping and the `ResidueSet` tokeniser to get the set of canonical UNIMOD residues it can produce,
I/L-collapsed (isoleucine and leucine are mass-equivalent and no de novo model can distinguish them).
Offending residues are tallied and logged, so "we dropped 12k rows" always comes with *which*
residues caused it.

Scoring a model against targets it cannot represent measures vocabulary coverage, not sequencing
ability — so those rows are removed for every model by the same code.

### Notation remapping

Casanovo detokenises to named-modification ProForma; the scorer works in UNIMOD:

```python
CASANOVO_TO_UNIMOD = {
    "C[Carbamidomethyl]": "C[UNIMOD:4]",
    "M[Oxidation]":       "M[UNIMOD:35]",
    "N[Deamidated]":      "N[UNIMOD:7]",
    "[Acetyl]":           "[UNIMOD:1]",
    "[+25.980265]":       "[UNIMOD:5][UNIMOD:385]",   # one token -> two residues
    …
}
```

`MODEL_SCORING["casanovo"]` carries it into `score_predictions`, which builds the `ResidueSet` with
that remapping. `_canonicalize_peptide` re-tokenises each mapped token and validates every residue it
produces — a remapping that expands to two mods is never a single vocab entry, so a naive membership
check would silently reject valid peptides. An unmappable residue yields `""`, i.e. a scored miss,
not a crash.

### Two denominators

`score_predictions` reports every metric twice:

| Suffix | Denominator | Use |
|--------|-------------|-----|
| *(none)* | **true** — all labelled rows; filtered and undecoded spectra count as misses | cross-model comparison |
| `_predictable` | **filtered** — only rows the model could attempt | "how good is it when it answers" |

The true denominator is the honest one. A model that discards a third of the spectra during
preprocessing and sequences the rest perfectly is not a better sequencer, and the filtered number
would say it was. Both are reported so the gap — visible as `n_unpredictable` — is explicit rather
than buried.

Targets whose residues are missing from the scoring residue set get a `_SENTINEL_MASS`
(1,000,000 Da + index) rather than being dropped, so they are scored as guaranteed misses. Dropping
them would quietly shrink the denominator in the model's favour.

### Outputs

```
de_novo_results/<run_name>/casanovo/
├── <dataset>.csv                      # canonical prediction CSV per dataset
├── <dataset>/summary.json             # overall (+ per-group) metric blocks
└── casanovo_de_novo_results.csv       # one aggregated row: <dataset>_<metric> per column
```

A local copy is always written. If a group's `output_path` (or `output_dir` / `result_file_path`) is
an `s3://` URI it is uploaded too; on failure or unconfigured S3 the local copy is kept and warned
about, unless `allow_local_fallback=false` makes it a hard error.
