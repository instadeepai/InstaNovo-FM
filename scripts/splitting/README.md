# Data Splitting Workflow Guide

This guide helps you reproduce the data splitting workflow for large-scale mass spectrometry datasets. All scripts use Typer for command-line interaction.

## Quick Start

### Basic Workflow

1. Build quality-curated subsets MCFM and HCFM from LCFM with `create_subsets.py`
2. Load the existing InstNovo peptide registry from HuggingFace (default) or a local copy via `--registry-dir`.
3. Update splits on labelled trees (typically LCFM first) so unseen peptides are assigned without moving known ones.
4. Split each corpus (LCFM, and MCFM/HCFM) with `split-only` against that registry.
5. Shuffle data for unbiased training.

### Example: Complete Workflow

```bash
# 1. Build MCFM / HCFM confidence subsets from scored labelled data
python create_subsets.py \
    --input-dir <data-root>/lcfm \
    --medium-output-dir <data-root>/mcfm \
    --high-output-dir <data-root>/hcfm

# 2. Extend the InstaNovo registry with new labelled peptides (LCFM),
#    then write train/test/valid parquet. Omit --registry-dir to download
#    peptide_registry.parquet from HuggingFace.
python split_labelled_data.py split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --mode both

# 3. Optionally update the registry from additional labelled trees in one pass
python split_labelled_data.py batch <data-root>/mcfm <data-root>/hcfm \
    --output-dir <data-root>/registry_update \
    --registry-dir <data-root>/lcfm_splits \
    --mode update-splits

# 4. Partition each corpus using the updated registry (errors if any peptide
#    is missing — run update-splits first for that tree)
python split_labelled_data.py split \
    --input-dir <data-root>/mcfm \
    --output-dir <data-root>/mcfm_splits \
    --registry-dir <data-root>/registry_update \
    --mode split-only
python split_labelled_data.py split \
    --input-dir <data-root>/hcfm \
    --output-dir <data-root>/hcfm_splits \
    --registry-dir <data-root>/registry_update \
    --mode split-only

# 5. Shuffle the splits
python shuffle_2pass.py --base-dir <data-root> --output-dir shuffled_splits
```

### About the peptide registry

Labelled splitting keeps peptides in the split they already had in the base InstaNovo HuggingFace registry (`peptide_registry.parquet`), then assigns only unseen sequences 80/10/10.
That is how you extend the training data without breaking consistency with the original model splits.

Peptide identity for the registry (and for leakage checks) is the unmodified peptide with I and L treated as the same residue:

1. Prefer the `unmodified_peptide` column when present. If it is missing, derive it from `sequence` by stripping bracketed and parenthesised modification tags (e.g. `[UNIMOD:4]`) and hyphens.
2. Map every `I` to `L` on that string. Registry keys and split assignment use this normalised form, so `PEPTIDE` and `PEPTLDE` share a split.

Quality filters (see Data Quality Requirements below) apply before both registry updates and writing split parquet.
Only spectra that pass are considered; failing rows are skipped (not written).

- Default: omit `--registry-dir` to download the registry from HuggingFace.
- Local copy: pass `--registry-dir` pointing at a directory that contains `peptide_registry.parquet` (e.g. the output of a previous `update-splits` run).
- Modes: `update-splits` only updates and saves the registry; `split-only` partitions files and fails if any peptide is not in the registry; `both` (default) does update then split.

Confidence subsets (MCFM/HCFM) are built separately with `create_subsets.py` (score thresholds), not by a second train/test/val splitter.

## Individual Script Usage

### 1. Split Labelled Data (`split_labelled_data.py`)

Creates train/test/validation splits from labelled peptide data using the peptide registry.

```bash
# Basic usage (download HF registry, update + split)
python split_labelled_data.py split

# Custom parameters
python split_labelled_data.py split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --rows-per-file 500000

# Update registry only
python split_labelled_data.py split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --mode update-splits

# Split files only (registry must already contain all peptides)
python split_labelled_data.py split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --registry-dir <data-root>/lcfm_splits \
    --mode split-only

# Batch: one combined update/split pass over several directories
python split_labelled_data.py batch <data-root>/lcfm <data-root>/mcfm <data-root>/hcfm \
    --output-dir <data-root>/combined_splits
```

Key features:

- Respects existing HF (or local) registry assignments
- 80/10/10 train/test/validation for new peptides only
- Filters: RT ≤ 10800 s, lower offset ≤ 300 Da, charge 0–7 inclusive, m/z ≤ 2000, and no modification unresolvable to UNIMOD (`[IN:<digits>]` in `sequence`)
- Normalises peptides for registry lookup (strip mods if needed, then I→L; see above)
- Batch processing for multiple datasets

### 2. Create MCFM / HCFM Subsets (`create_subsets.py`)

Builds medium- and high-confidence parquet trees from scored PSM tables by a global score threshold.
This is not train/test/val splitting — use `split_labelled_data.py` afterward to partition those trees against the registry.

See `scripts/preprocessing/README.md` for scoring details and CLI usage.

### 3. ACFM Split Data (`split_unlabelled_data.py`)

Advanced splitting for unlabelled data using LSH clustering.

```bash
# Full processing (recommended)
python split_unlabelled_data.py \
    --mode full \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits

# LSH computation only
python split_unlabelled_data.py \
    --mode lsh_only \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits

# Splitting only (using existing LSH)
python split_unlabelled_data.py \
    --mode split_only \
    --input-dir /path/to/acfm/data \
    --output-dir ./acfm_splits \
    --lsh-assignments ./acfm_splits/lsh_assignments.parquet
```

Key features:

- LSH clustering for similarity-based grouping
- Checkpointing for long-running processes
- Memory-efficient processing
- Configurable LSH parameters

Checkpointing support:

- Automatic resume: restart with the same parameters to resume from the last checkpoint
- Checkpoint files: `split_checkpoint.json`, `split_buffers.pkl`, `split_progress.json`
- Checkpoint interval: save every N files (default: 10, configurable with `--checkpoint-interval`)
- Error recovery: automatically saves state before errors, allows resume after fixes
- Fresh start: use `--clear-checkpoint` to start over or `--no-checkpoint` to disable

### 4. 2-Pass Shuffle (`shuffle_2pass.py`)

High-performance shuffling for large datasets (such as ACFM and LCFM).

```bash
# Basic usage (auto-detects optimal settings)
python shuffle_2pass.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits

# Custom configuration
python shuffle_2pass.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits \
    --target-chunk-size 100000 \
    --first-pass-processes 8 \
    --second-pass-processes 8 \
    --seed 42

# Memory forecasting
python shuffle_2pass.py \
    --forecast-chunk-size \
    --available-ram-gb 32
```

Key features:

- Memory-efficient 2-pass algorithm
- Parallel processing
- Auto-detection of optimal chunk sizes
- Deterministic output with seeds

### 5. RAM-Based Shuffle (`shuffle_in_ram.py`)

Note: this approach was used for shuffling the HCFM and MCFM subsets.

Simple and fast shuffling for the smallest datasets that fit entirely in RAM.

```bash
# Basic usage
python shuffle_in_ram.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits

# Custom configuration
python shuffle_in_ram.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits \
    --target-chunk-size 50000 \
    --seed 42
```

Key features:

- Loads entire split into RAM for global shuffling
- Fastest approach for small datasets
- Auto-detects optimal chunk size from input files
- Deterministic output with seeds

When to use:

- Small datasets: when all data for a split fits in available RAM
- Subset datasets: like HCFM and MCFM, which are smaller than the main datasets
- Fast processing: when speed is more important than memory efficiency
- Simple workflow: when you want the most straightforward shuffling approach

### Alternative: Index-Based Shuffle (`shuffle_indices.py`)

Note: this is not the approach used in our workflow, but is available as a simpler alternative.

Slower but simpler shuffling approach for smaller datasets that still do not fit into RAM.

```bash
# Basic usage
python shuffle_indices.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits \
    --chunk-size 400000 \
    --seed 42
```

Key features:

- Simpler algorithm (easier to understand)
- Lower memory usage per process
- Index-based row selection
- Good for smaller datasets

### Shuffling Algorithm Comparison

| Approach | Time Complexity | Memory Complexity | Use Case |
| --- | --- | --- | --- |
| RAM-Based | O(N) | O(N) total | Smallest datasets that fit in RAM (HCFM, MCFM) |
| 2-Pass | O(N + M log M) | O(chunk_size × processes) total | Extremely large datasets where time is a constraint |
| Index-Based | O(N²) | O(N) for indices + O(chunk_size) single-threaded | Medium datasets where memory is a constraint |

## Typer CLI Benefits

All scripts use Typer for an enhanced CLI experience:

- Auto-generated help: `python script.py --help`
- Command completion: tab completion for options
- Type validation: automatic type checking
- Rich error messages: clear error reporting

### Example: Getting Help

```bash
# Show all commands
python split_labelled_data.py --help

# Show command-specific help
python split_labelled_data.py split --help

# Show batch command help
python split_labelled_data.py batch --help
```

## Data Format Requirements

### Input Files

- Parquet files: for split scripts (peptide data)
- Directory structure: organised by dataset type

### Output Files

- Split parquet files: organised by train/test/validation
- Peptide registry: `peptide_registry.parquet` written under `--output-dir` on update

### Required Columns and Datatypes

#### For Split Scripts (Parquet Input)

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

#### For ACFM Script (Additional Columns)

| Column | Datatype | Description |
| --- | --- | --- |
| `modified_peptide` | `str` | Modified peptide sequence |
| `modifications` | `str` | Modification information |

### Data Quality Requirements

Filtering criteria applied — these are `filter_spectra()` in `split_labelled_data.py` (and the same predicates in `collect_unique_peptides`); five conditions, and a row is kept only if it satisfies all of them:

- `retention_time` ≤ 10800 seconds
- `lower_offset` ≤ 300 Da
- `precursor_charge` ≥ 0 and ≤ 7 — note 0 is kept, not excluded
- `precursor_mz` ≤ 2000 Da
- `sequence` contains no `[IN:<digits>]` token — a private namespace for modifications the pipeline could not resolve to a UNIMOD identifier. Measured across the corpus these are 175 distinct tokens, predominantly glycopeptides.

Rows that fail are omitted from the registry update and from the split outputs; input parquet is left unchanged.
