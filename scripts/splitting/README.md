# Splitting scripts

Build confidence subsets, peptide-disjoint train/test/validation splits, and shuffled shards.
For the full pipeline order, which runs preprocessing then verification then this stage, see [`../README.md`](../README.md).

All scripts in this folder use Typer.
Run them from the repository root as `uv run python -m scripts.splitting.<script>` (see [`../README.md`](../README.md)).

## Basic workflow

1. Build quality-curated subsets MCFM and HCFM from LCFM with `create_subsets.py`.
2. Load the existing InstNovo peptide registry from HuggingFace (default) or a local copy via `--registry-dir`.
3. Update splits on labelled trees (typically LCFM first) so unseen peptides are assigned without moving known ones.
4. Split each corpus (LCFM, and MCFM/HCFM) with `split-only` against that registry.
5. Shuffle data for unbiased training.

```bash
# 1. Build MCFM / HCFM confidence subsets from scored labelled data
uv run python -m scripts.splitting.create_subsets \
    --input-dir <data-root>/lcfm \
    --medium-output-dir <data-root>/mcfm \
    --high-output-dir <data-root>/hcfm

# 2. Extend the InstaNovo registry with new labelled peptides (LCFM),
#    then write train/test/valid parquet. Omit --registry-dir to download
#    peptide_registry.parquet from HuggingFace.
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --mode both

# 3. Optionally update the registry from additional labelled trees in one pass
uv run python -m scripts.splitting.split_labelled_data batch <data-root>/mcfm <data-root>/hcfm \
    --output-dir <data-root>/registry_update \
    --registry-dir <data-root>/lcfm_splits \
    --mode update-splits

# 4. Partition each corpus using the updated registry (errors if any peptide
#    is missing — run update-splits first for that tree)
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/mcfm \
    --output-dir <data-root>/mcfm_splits \
    --registry-dir <data-root>/registry_update \
    --mode split-only
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/hcfm \
    --output-dir <data-root>/hcfm_splits \
    --registry-dir <data-root>/registry_update \
    --mode split-only

# 5. Shuffle the splits
uv run python -m scripts.splitting.shuffle_2pass --input-dir <data-root>/lcfm_splits --output-dir shuffled_splits
```

### About the peptide registry

Labelled splitting keeps peptides in the split they already had in the base InstaNovo HuggingFace registry (`peptide_registry.parquet`), then assigns only unseen sequences 80/10/10.
That is how you extend the training data without breaking consistency with the original model splits.

Peptide identity for the registry (and for leakage checks) is the unmodified peptide with I and L treated as the same residue:

1. Prefer the `unmodified_peptide` column when present. If it is missing, derive it from `sequence` by stripping bracketed and parenthesised modification tags such as `[UNIMOD:4]`, along with hyphens.
2. Map every `I` to `L` on that string. Registry keys and split assignment use this normalised form, so `PEPTIDE` and `PEPTLDE` share a split.

Quality filters (see data quality requirements below) apply before both registry updates and writing split parquet.
Only spectra that pass are considered; failing rows are skipped (not written).

- Default: omit `--registry-dir` to download the registry from HuggingFace.
- Local copy: pass `--registry-dir` pointing at a directory that contains `peptide_registry.parquet` (e.g. the output of a previous `update-splits` run).
- Modes: `update-splits` only updates and saves the registry; `split-only` partitions files and fails if any peptide is not in the registry; `both` (default) does update then split.

Confidence subsets (MCFM/HCFM) are built separately with `create_subsets.py` (score thresholds), not by a second train/test/val splitter.

## Individual script usage

### 1. Split labelled data (`split_labelled_data.py`)

Creates train/test/validation splits from labelled peptide data using the peptide registry.

```bash
# Basic usage (download HF registry, update + split)
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits

# Custom parameters
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --rows-per-file 500000

# Update registry only
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --mode update-splits

# Split files only (registry must already contain all peptides)
uv run python -m scripts.splitting.split_labelled_data split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --registry-dir <data-root>/lcfm_splits \
    --mode split-only

# Batch: one combined update/split pass over several directories
uv run python -m scripts.splitting.split_labelled_data batch <data-root>/lcfm <data-root>/mcfm <data-root>/hcfm \
    --output-dir <data-root>/combined_splits
```

**Key features:**

- Respects existing HF (or local) registry assignments
- 80/10/10 train/test/validation for new peptides only
- Filters: RT ≤ 10800 s, lower offset ≤ 300 Da, charge 0–7 inclusive, m/z ≤ 2000, and no modification unresolvable to UNIMOD (`[IN:<digits>]` in `sequence`)
- Normalises peptides for registry lookup (strip mods if needed, then I→L; see above)
- Batch processing for multiple datasets

### 2. Create MCFM / HCFM subsets (`create_subsets.py`)

Builds medium-confidence (MCFM) and high-confidence (HCFM) parquet trees from a root directory of all-confidence labelled spectra, with one subfolder per dataset and `.parquet` files inside.
This is separate from `split_labelled_data.py`, because it filters rows by a global score threshold rather than by train, test and validation splits.

#### How scoring works

1. The composite score, held in the temporary column `_composite_score`, combines EncyclopeDIA-style scores, each taken as a percentile rank within peptide length.
   - `1 - expectation`, `probability` and `hyperscore` contribute ranked components.
   - If `nextscore` is present, the term `(hyperscore - nextscore)` is included and the average is over four parts rather than three.
   - Rank uses `(rank - 1) / (count - 1)`, so the best item in a group scores 1.0, the worst scores 0.0, and singletons default to 0.5.
   - Peptide length comes from a `peptide_length` or `peptide length` column when present, and otherwise from the uppercase-letter count in `peptide` or `unmodified_peptide`.
2. Pass one reads every file, computes composite scores for the finite values, and concatenates them.
3. Global thresholds are then taken as linear quantiles on the pooled scores.
   - MCFM keeps rows scoring strictly above the 90th percentile, roughly the top 10% of PSMs globally.
   - HCFM keeps rows scoring strictly above the 98th percentile, roughly the top 2% globally.
4. Pass two applies those thresholds and writes the filtered parquets to `--medium-output-dir` and `--high-output-dir`, preserving the input subfolder and filename layout.

Passing `--hold-back-modified-rows` drops rows whose `sequence` contains the internal `[IN:<digits>]` modification tokens before scoring and output, so those rows affect neither the thresholds nor the exported subsets.

```bash
uv run python -m scripts.splitting.create_subsets \
    --input-dir <data-root>/lcfm/ \
    --medium-output-dir <data-root>/mcfm/ \
    --high-output-dir <data-root>/hcfm/ \
    --hold-back-modified-rows
```

### 3. ACFM split data (`split_unlabelled_data.py`)

Advanced splitting for unlabelled data using LSH clustering.

```bash
# Full processing (recommended)
uv run python -m scripts.splitting.split_unlabelled_data \
    --mode full \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits

# LSH computation only
uv run python -m scripts.splitting.split_unlabelled_data \
    --mode lsh_only \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits

# Splitting only (using existing LSH)
uv run python -m scripts.splitting.split_unlabelled_data \
    --mode split_only \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits \
    --lsh-assignments ./acfm_splits/lsh_assignments.parquet
```

**Key features:**

- LSH clustering for similarity-based grouping
- Checkpointing for long-running processes
- Memory-efficient processing
- Configurable LSH parameters

**Checkpointing support:**

- Automatic resume: restart with the same parameters to resume from the last checkpoint
- Checkpoint files: `split_checkpoint.json`, `split_buffers.pkl`, `split_progress.json`
- Checkpoint interval: save every N files (default: 10, configurable with `--checkpoint-interval`)
- Error recovery: automatically saves state before errors, allows resume after fixes
- Fresh start: use `--clear-checkpoint` to start over or `--no-checkpoint` to disable

### 4. 2-pass shuffle (`shuffle_2pass.py`)

High-performance shuffling for large datasets (such as ACFM and LCFM).

```bash
# Basic usage (auto-detects optimal settings)
uv run python -m scripts.splitting.shuffle_2pass \
    --input-dir <data-root>/lcfm_splits \
    --output-dir shuffled_splits

# Custom configuration
uv run python -m scripts.splitting.shuffle_2pass \
    --input-dir <data-root>/lcfm_splits \
    --output-dir shuffled_splits \
    --chunk-size 100000 \
    --pass1-procs 8 \
    --pass2-procs 8 \
    --seed 42

# Memory forecasting
uv run python -m scripts.splitting.shuffle_2pass \
    --forecast \
    --ram-gb 32 \
    --sample-file train_0.parquet
```

**Key features:**

- Memory-efficient 2-pass algorithm
- Parallel processing
- Auto-detection of optimal chunk sizes
- Deterministic output with seeds

### 5. RAM-based shuffle (`shuffle_in_ram.py`)

Note: this approach was used for shuffling the HCFM and MCFM subsets.

Simple and fast shuffling for the smallest datasets that fit entirely in RAM.

```bash
# Basic usage
uv run python -m scripts.splitting.shuffle_in_ram \
    --input-dir <data-root>/hcfm_splits \
    --output-dir shuffled_splits \
    --target-chunk-size 50000

# Custom configuration
uv run python -m scripts.splitting.shuffle_in_ram \
    --input-dir <data-root>/hcfm_splits \
    --output-dir shuffled_splits \
    --target-chunk-size 50000 \
    --seed 42
```

**Key features:**

- Loads entire split into RAM for global shuffling
- Fastest approach for small datasets
- Auto-detects optimal chunk size from input files
- Deterministic output with seeds

**When to use:**

- Small datasets: when all data for a split fits in available RAM
- Subset datasets: like HCFM and MCFM, which are smaller than the main datasets
- Fast processing: when speed is more important than memory efficiency
- Simple workflow: when you want the most straightforward shuffling approach

### Alternative: index-based shuffle (`shuffle_indices.py`)

Note: this is not the approach used in our workflow, but is available as a simpler alternative.

Slower but simpler shuffling approach for smaller datasets that still do not fit into RAM.

```bash
# Basic usage
uv run python -m scripts.splitting.shuffle_indices \
    --input-dir <data-root> \
    --output-dir shuffled_splits \
    --chunk-size 400000 \
    --seed 42
```

**Key features:**

- Simpler algorithm (easier to understand)
- Lower memory usage per process
- Index-based row selection
- Good for smaller datasets

### Shuffling algorithm comparison

| Approach | Time complexity | Memory complexity | Use case |
| --- | --- | --- | --- |
| RAM-based | O(N) | O(N) total | Smallest datasets that fit in RAM (HCFM, MCFM) |
| 2-pass | O(N + M log M) | O(chunk_size × processes) total | Extremely large datasets where time is a constraint |
| Index-based | O(N²) | O(N) for indices + O(chunk_size) single-threaded | Medium datasets where memory is a constraint |

## Data requirements

### Input files

- Parquet files: for split scripts (peptide data)
- Directory structure: organised by dataset type

### Output files

- Split parquet files: organised by train/test/validation
- Peptide registry: `peptide_registry.parquet` written under `--output-dir` on update

### Required columns and datatypes

#### Labelled LCFM, MCFM and HCFM (`split_labelled_data.py`)

| Column | Datatype | Description |
| --- | --- | --- |
| `unmodified_peptide` or `sequence` | `str` | Peptide sequence (e.g., "ACDEFGHIK") |
| `retention_time` | `float` | Retention time in seconds |
| `precursor_mz` | `float` | Precursor m/z value |
| `precursor_charge` | `int` | Precursor charge state (0-7; 0 denotes an unassigned charge) |
| `lower_offset` | `float` | Lower offset value |
| `collision_energy` | `str` | Collision energy (can be "Unknown") |
| `isolation_target` | `float` | Isolation target m/z |
| `frag_type` | `str` | Fragmentation type |
| `mz_array` | `list[float]` | m/z array for spectrum |
| `intensity_array` | `list[float]` | Intensity array for spectrum |
| `scan` | `int` | Scan number |
| `header` | `str` | Experiment header |
| `index` | `int` | Row index |
| `scale_factor` | `float` | Intensity scaling factor |

#### Unlabelled ACFM (`split_unlabelled_data.py`)

Spectrum columns match the labelled table (`mz_array`, `intensity_array`, and the other scan-level fields). These two peptide fields are extra:

| Column | Datatype | Description |
| --- | --- | --- |
| `modified_peptide` | `str` | Modified peptide sequence |
| `modifications` | `str` | Modification information |

### Data quality requirements

Filtering criteria applied — these are `filter_spectra()` in `split_labelled_data.py` (and the same predicates in `collect_unique_peptides`); five conditions, and a row is kept only if it satisfies all of them:

- `retention_time` ≤ 10800 seconds
- `lower_offset` ≤ 300 Da
- `precursor_charge` ≥ 0 and ≤ 7 — note 0 is kept, not excluded
- `precursor_mz` ≤ 2000 Da
- `sequence` contains no `[IN:<digits>]` token — a private namespace for modifications the pipeline could not resolve to a UNIMOD identifier. Measured across the corpus these are 175 distinct tokens, predominantly glycopeptides.

Rows that fail are omitted from the registry update and from the split outputs; input parquet is left unchanged.
