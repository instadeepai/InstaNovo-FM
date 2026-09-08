# Baseline embedding benchmark — duplicate retrieval, linear probe, UMAP

Runs the **encoder** of InstaNovo, Casanovo and XuanjiNovo over our datasets, saves the pooled
spectrum embeddings, and scores them with the *same* task code and the *same* config the
foundation model is scored with.

Two entry points, both env-var driven:

| Script | Tasks | Default |
|--------|-------|---------|
| [`scripts/run_baseline_eval.sh`](../../../scripts/run_baseline_eval.sh) | `linearprobetask`, `duplicateretrievaltask` | quantitative metrics |
| [`scripts/run_baseline_umaps.sh`](../../../scripts/run_baseline_umaps.sh) | `umapvisualisationtask`, `evocclusteringtask` | figures |

Both are the same two-stage shape:

```
dataset config ──> extract_<model>_embeddings ──> embeddings.h5 + index.faiss
                                                        │
                                                        └──> run_tasks_from_embeddings ──> task_results.json
```

Stage 2 needs no model, so figures and metrics can be re-rendered from a saved embedding store at
any time.

---

## 1. What makes the numbers comparable

Four things are read from config rather than hardcoded, so a baseline can never silently drift from
the FM run it is being compared against:

| Value | Source of truth | Read by |
|-------|-----------------|---------|
| parquet paths per split | `configs/dataset/<EVAL_DATASET>.yaml` | `_load_dataset_paths` |
| `max_mz` (m/z normalisation divisor) | `configs/model/foundation_base.yaml` | `_load_foundation_max_mz` |
| probe split sizes (100k/10k/10k) | `configs/evaluation/<EVAL_CONFIG_NAME>.yaml` | `_load_eval_task_configs` |
| every task setting (k-values, conditional subsets, probe targets, …) | same file, `task_configs` block | `--eval_config_name` |

`--eval_config_name` is the important one: it loads the canonical `task_configs` block as the
**base** configuration and deep-merges `--task_configs` JSON on top. There is no hand-written task
JSON to fall out of sync with the FM protocol.

By default every model is also fed the **same** spectra through the **same**
`FoundationalDataProcessor` (200 peaks, 50–2500 m/z, `min_intensity=0.01`, m/z divided by `max_mz`).
Only the encoder differs.

---

## 2. Models and pooling

`MODEL` selects the extractor, the canonical checkpoint and the download fallback:

| `MODEL` | Extractor | Checkpoint | Encoder token layout | Pooling options |
|---------|-----------|------------|----------------------|-----------------|
| `casanovo` | `extract_casanovo_embeddings` | `casanovo_v5_0_0.ckpt` (GitHub release) | `[cls, peak_1 … peak_N]` | `cls`, `mean_peaks` |
| `instanovo` | `extract_instanovo_embeddings` | `instanovo_v1_2_0.ckpt` (GitHub release) | `[precursor, latent, peak_1 … peak_N]` | `latent`, `mean_peaks` |
| `xuanjinovo` | `extract_xuanjinovo_embeddings` | `XuanjiNovo_100M_massnet.ckpt` (HF) | `[precursor, peak_1 … peak_N]` | `precursor`, `mean_peaks` |
| `fm` | (short-circuits to `embed_evaluation`) | required | — | set by eval config |
| `fm_de_novo` | `extract_fm_de_novo_embeddings` | required | `[latent, peak_1 … peak_N]` | `mean_peaks`, `last_token` |

Every extractor asserts its expected token count (`n_peaks + 1` or `+ 2`) on every batch, so an
upstream layout change fails loudly instead of pooling the wrong tokens.

**`POOLING=mean_peaks` is the default and the comparable choice.** Each model's native summary token
is trained for a different job (Casanovo's `cls` feeds a decoder, InstaNovo's `latent` is its
designed spectrum summary, XuanjiNovo mixes the precursor into position 0), so comparing native
tokens compares training objectives as much as representations. Mean-over-peak-tokens is the one
pooling every architecture supports identically, and it matches the FM's `mean_pool`.

`MODEL=fm_de_novo` evaluates the **encoder of a downstream de novo checkpoint** (FM encoder +
InstaNovo decoder — see [the downstream README](../downstream/de_novo_sequencing/README.md)) through
this same path. It has no canonical checkpoint, so `CHECKPOINT` is required.

XuanjiNovo is loaded through `_xuanjinovo_encoder.py`, a vendored numerically-unchanged copy of the
upstream spectrum encoder — upstream pins `torch==2.1` and a C++ `ctcdecode` extension that cannot
coexist with this environment. The vendored copy covers the **encoder only**; upstream de novo
decoding needs the separate Docker route in [`README_xuanjinovo.md`](README_xuanjinovo.md).

---

## 3. Stage 1 — extraction

```bash
MODEL=instanovo EVAL_DATASET=lcfm bash scripts/run_baseline_eval.sh
```

Each extraction writes one embedding store:

```
embed_eval_results/instanovo/
├── valid/                     # the SPLIT pool: duplicate retrieval, UMAP, EVōC
│   ├── embeddings.h5          # (N, D) float32 + a metadata group + embedding_pooling attr
│   └── index.faiss            # IndexFlatIP over L2-normalised rows == cosine
├── probe_train/               # only extracted when linearprobetask is requested
├── probe_val/
└── probe_test/
```

Metadata travels with the embeddings, which is what lets stage 2 run model-free. Three sources feed
it (`_extract_embeddings_common.collect_batch_metadata` → `_extract_metadata_common.finalize_metadata`):

1. **From the batch** — `precursor_mz`, `precursor_charge`, `precursor_mass`, `peptides`/`sequence`,
   `frag_type`, `collision_energy`, `usi`, `header`, `search_*` columns present in the parquet.
2. **From `search_data.csv` by USI lookup** — `search_instrument`, `search_project`,
   `search_detector`, `search_organism`. Match counts are logged per field.
3. **Derived from the peptide** — `modification_types`, `ptm_present`, `modification_class` (top-6
   classes, rest collapsed to `Other`), `hydrophobicity`.

`frag_type` is deliberately **never** overwritten by the CSV lookup: the FM reads fragmentation
straight from the parquet, so the baseline keeps the parquet value too.


---

## 4. Stage 2 — the metric tasks

### Duplicate retrieval

Groups the pool by identical peptide sequence, queries each member against the FAISS index (dropping
the self-hit), and reports:

| Metric | Meaning |
|--------|---------|
| `recall@k` | binary — did *any* same-peptide spectrum land in the top k |
| `prop_recall@k` | fraction of the group's other members found in the top k |
| `map@20` | ranking quality (AP normalised by `min(n_relevant, n_retrieved)`) |

### Linear probe

Trains a linear model per target on standardised embeddings — logistic regression for categorical
targets, ridge for continuous, auto-detected. cuML is used when available, with a logged fallback to sklearn.

The default targets are the ten in `configs/evaluation/default.yaml`:

```
precursor_charge, precursor_mz, precursor_mass, hydrophobicity, ptm_present,
modification_class, search_instrument, frag_type, collision_energy, spectrum_confidence
```

**Split protocol.** `run_baseline_eval.sh` always passes `--probe_split_dirs`, which puts the probe
in `pre_filtered` mode: train on the `probe_train` store, tune on `probe_val`, report on
`probe_test`. This is the FM evaluator's protocol, and the caps are read from the same eval config,
so the probe sees the same amount of the same data as the FM did.

---

## 5. UMAP and EVōC figures

```bash
MODEL=casanovo EVAL_DATASET=lcfm MAX_SAMPLES=100000 POOLING=mean_peaks bash scripts/run_baseline_umaps.sh
```

Same stage 1, then two model-free figure tasks. `MAX_SAMPLES` is forwarded into both task configs
(overriding the config default of 20k) and the EVōC zoom-recolor panel is switched on, so the output
matches the original FM figures.

**`umapvisualisationtask`** projects to 2D (`n_neighbors=30`, `min_dist=0.1`, cosine) and renders one
PNG per metadata field it can colour — `umap_frag_type.png`, `umap_search_instrument.png`,
`umap_search_organism.png`, … plus `umap_summary_panel.png`. Rare categories collapse into `Other`
past `max_categories=15`.
It also writes `umap_coordinates.npz` and `umap_coordinates.parquet` — the 2D coordinates plus every
per-spectrum metadata column. Zoomed or recoloured figures can then be regenerated offline without
re-running the encoder. Each save is independently wrapped so a single bad metadata column cannot
kill the figure run.

---

## 6. Single-`data_path` datasets

When a dataset config defines a single `data_path` instead of `train_path`/`valid_path`/`test_path`,
`run_baseline_eval.sh` calls `split_data_path` to partition it into **sequence-disjoint**
train/valid/test on the fly (sorted unique sequences, fixed `SPLIT_SEED=42`, `SPLIT_GROUP_KEY=sequence`),
writing them under `${OUTPUT_DIR}/_splits`. Sequence-disjointness matters here: the probe would
otherwise be scored on peptides it trained on, and duplicate retrieval would be measuring leakage.

---

## 7. Parameters

Both scripts take everything from the environment. The ones that change results:

| Var | Default | Note |
|-----|---------|------|
| `MODEL` | *(required)* | see the table in §2 |
| `EVAL_DATASET` | `lcfm` | literal filename of `configs/dataset/<name>.yaml`; use `mcfm_local` locally |
| `SPLIT` | `valid` | pool for the single-split tasks; probe splits are separate and unaffected |
| `TASKS` | `linearprobetask` | **space**-separated (eval script only) |
| `DUP_MAX_SAMPLES` | `200000` | pool size — must equal the FM run's `evaluation.max_samples` |
| `MAX_SAMPLES` | `100000` | umap script's equivalent |
| `POOLING` | `mean_peaks` | see §2 |
| `PREPROCESSING` | `foundation` | `native` only for casanovo/xuanjinovo; tags the output dir |
| `MIN_INTENSITY` | extractor default (0.01) | tags the output dir `_mi<value>` |
| `EVAL_CONFIG_NAME` | `default` | `default_local` for laptop runs |
| `CHECKPOINT` | per-model canonical | local path, http(s) URL, or `s3://…` (cached under `checkpoints/`) |
| `DEVICE` / `BATCH_SIZE` / `NUM_WORKERS` | `cuda` / `256` / `4` | |

`s3://` checkpoints resolve through `S3FileHandler` and need `AWS_ENDPOINT_URL`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (already set on AIchor; from `.env` locally). They are
cached under `checkpoints/` so they download once.
