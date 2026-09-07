# Data Splitting Workflow Guide

> **Ported from the internal repo.** Paths that were mount-specific are written
> as `<data-root>`; pass your own. The peptide registry is fetched from and
> published to the HuggingFace dataset (see `split_labelled_data.py`).

This guide helps you reproduce the data splitting workflow for large-scale mass spectrometry datasets. All scripts use **Typer CLI** for easy command-line interaction.

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Workflow
1. **Consolidate splits** from multiple sources
2. **Create initial splits** from labeled data (lcfm)
3. **Create subset splits** using existing assignments (mcfm, hcfm)
4. **Shuffle data** for unbiased training

### Example: Complete Workflow
```bash
# 1. Consolidate multiple split files first
python create_consolidated_split_assignments.py consolidate splits/file1.csv splits/file2.csv

# 2. Split labeled data (lcfm)
python split_labelled_data.py split --input-dir <data-root>/lcfm --output-dir <data-root>/lcfm_splits

# 3. Create subset splits (mcfm, hcfm)
python create_labelled_subset_splits.py subset --input-dir <data-root>/mcfm --output-dir <data-root>/mcfm_splits
python create_labelled_subset_splits.py subset --input-dir <data-root>/hcfm --output-dir <data-root>/hcfm_splits

# 4. Shuffle the splits
python shuffle_2pass.py --base-dir <data-root> --output-dir shuffled_splits
```

### About Consolidation

The consolidation step combines existing split files from datasets used to train the base InstaNovo model. This allows you to add new datasets (lcfm, mcfm, hcfm) to the existing training data while maintaining consistency with the original splits.

**When to use consolidation:**
- **Adding to existing model**: If you're extending a pre-trained InstaNovo model
- **Consistency requirements**: When you need to maintain split assignments from previous training
- **Incremental training**: For fine-tuning or transfer learning scenarios

**When consolidation is optional:**
- **New model training**: If you're training a model from scratch
- **Independent datasets**: When your new datasets are self-contained
- **Research experiments**: When you want fresh, independent splits

If you're not extending an existing model, you can skip the consolidation step and proceed directly to splitting your datasets.

## Individual Script Usage

### 1. Split Labeled Data (`split_labelled_data.py`)

Creates train/test/validation splits from labeled peptide data.

```bash
# Basic usage
python split_labelled_data.py split

# Custom parameters
python split_labelled_data.py split \
    --input-dir <data-root>/lcfm \
    --output-dir <data-root>/lcfm_splits \
    --rows-per-file 500000

# Batch processing
python split_labelled_data.py batch <data-root>/lcfm <data-root>/mcfm <data-root>/hcfm \
    --output-base <data-root>
```

**Key Features:**
- 80/10/10 train/test/validation split ratios
- Filters: RT ≤ 10800 s, lower offset ≤ 300 Da, charge 0–7 **inclusive**, m/z ≤ 2000,
  and no modification unresolvable to UNIMOD (`[IN:<digits>]` in `sequence`)
- Peptide-level shuffling with split preservation
- Batch processing for multiple datasets

### 2. Create Subset Splits (`create_labelled_subset_splits.py`)

Creates splits for subset datasets using existing assignments.

```bash
# Basic usage
python create_labelled_subset_splits.py subset

# Custom parameters
python create_labelled_subset_splits.py subset \
    --input-dir <data-root>/hcfm \
    --output-dir <data-root>/hcfm_splits \
    --split-assignments <data-root>/output_files/split_assignments.csv

# Batch processing
python create_labelled_subset_splits.py batch <data-root>/mcfm <data-root>/hcfm \
    --output-base <data-root>
```

**Key Features:**
- Uses existing split assignments from main dataset
- Same filtering criteria as main split script
- Maintains consistency across datasets

### 3. Consolidate Splits (`create_consolidated_split_assignments.py`)

Combines multiple split files and resolves conflicts.

```bash
# Use default files
python create_consolidated_split_assignments.py default

# Custom consolidation
python create_consolidated_split_assignments.py consolidate \
    splits/file1.csv splits/file2.csv \
    --output-file splits/consolidated.csv

# Batch processing
python create_consolidated_split_assignments.py batch "splits/*_splits.csv"
```

**Key Features:**
- Handles I/L ambiguity normalization
- Creates blacklist for conflicting assignments
- Supports glob patterns for batch processing

### 4. ACFM Split Data (`split_unlabelled_data.py`)

Advanced splitting for unlabeled data using LSH clustering.

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

**Key Features:**
- LSH clustering for similarity-based grouping
- Checkpointing for long-running processes
- Memory-efficient processing
- Configurable LSH parameters

**Checkpointing Support:**
- **Automatic resume**: Restart with same parameters to resume from last checkpoint
- **Checkpoint files**: `split_checkpoint.json`, `split_buffers.pkl`, `split_progress.json`
- **Checkpoint interval**: Save every N files (default: 10, configurable with `--checkpoint-interval`)
- **Error recovery**: Automatically saves state before errors, allows resume after fixes
- **Fresh start**: Use `--clear-checkpoint` to start over or `--no-checkpoint` to disable

### 5. 2-Pass Shuffle (`shuffle_2pass.py`)

High-performance shuffling for large datasets.

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

**Key Features:**
- Memory-efficient 2-pass algorithm
- Parallel processing
- Auto-detection of optimal chunk sizes
- Deterministic output with seeds

### 6. RAM-Based Shuffle (`shuffle_in_ram.py`)

**Note: This approach was used for shuffling the hcfm and mcfm subsets.**

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

**Key Features:**
- Loads entire split into RAM for global shuffling
- Fastest approach for small datasets
- Auto-detects optimal chunk size from input files
- Deterministic output with seeds

**When to use:**
- **Small datasets**: When all data for a split fits in available RAM
- **Subset datasets**: Like hcfm and mcfm which are smaller than main datasets
- **Fast processing**: When speed is more important than memory efficiency
- **Simple workflow**: When you want the most straightforward shuffling approach

### Alternative: Index-Based Shuffle (`shuffle_indices.py`)

**Note: This is NOT the approach used in our workflow, but available as a simpler alternative.**

Slower but simpler shuffling approach for smaller datasets that still do not fit into RAM.

```bash
# Basic usage
python shuffle_indices.py \
    --base-dir <data-root> \
    --output-dir shuffled_splits \
    --chunk-size 400000 \
    --seed 42
```

**Key Features:**
- Simpler algorithm (easier to understand)
- Lower memory usage per process
- Index-based row selection
- Good for smaller datasets

### Shuffling Algorithm Comparison

| Approach | Time Complexity | Memory Complexity | Use Case |
|----------|----------------|-------------------|----------|
| **RAM-Based** | O(N) | O(N) total | Smallest datasets that fit in RAM (hcfm, mcfm) |
| **2-Pass** | O(N + M log M) | O(chunk_size × processes) total | Extremely large datasets where time is a constraint |
| **Index-Based** | O(N²) | O(N) for indices + O(chunk_size) single-threaded | Medium datasets where memory is a constraint |

## Typer CLI Benefits

All scripts use **Typer** for enhanced CLI experience:

- **Auto-generated help**: `python script.py --help`
- **Command completion**: Tab completion for options
- **Type validation**: Automatic type checking
- **Rich error messages**: Clear error reporting
- **Legacy compatibility**: Original functions preserved

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
- **Parquet files**: For split scripts (peptide data)
- **CSV files**: For consolidate script (sequence + split columns)
- **Directory structure**: Organized by dataset type

### Output Files
- **Split parquet files**: Organized by train/test/validation
- **Consolidated CSV**: Resolved conflicts and assignments
- **Blacklist files**: Peptides to exclude from processing

### Required Columns and Datatypes

#### For Split Scripts (Parquet Input)
| Column | Datatype | Description | Required |
|--------|----------|-------------|----------|
| `peptide` | `str` | Peptide sequence (e.g., "ACDEFGHIK") | Yes |
| `rt` | `float` | Retention time in seconds | Yes |
| `precursor_mz` | `float` | Precursor m/z value | Yes |
| `charge` | `int` | Precursor charge state (0-7; 0 denotes an unassigned charge) | Yes |
| `offset` | `float` | Lower offset value | Yes |
| `collision_energy` | `str` | Collision energy (can be "Unknown") | Yes |
| `isolation_target` | `float` | Isolation target m/z | Yes |
| `frag_type` | `str` | Fragmentation type | Yes |
| `mz` | `list[float]` | m/z array for spectrum | Yes |
| `intensity` | `list[float]` | Intensity array for spectrum | Yes |
| `scan` | `int` | Scan number | Yes |
| `header` | `str` | Experiment header | Yes |
| `index` | `int` | Row index | Yes |
| `scale_factor` | `float` | Intensity scaling factor | Yes |

#### For Consolidate Script (CSV Input)
| Column | Datatype | Description | Required |
|--------|----------|-------------|----------|
| `sequence` | `str` | Peptide sequence | Yes |
| `split` | `str` | Split assignment ("train", "test", "valid") | Yes |

#### For ACFM Script (Additional Columns)
| Column | Datatype | Description | Required |
|--------|----------|-------------|----------|
| `modified_peptide` | `str` | Modified peptide sequence | Yes |
| `modifications` | `str` | Modification information | Yes |

### Data Quality Requirements

**Filtering Criteria Applied** — these are `filter_spectra()` in
`split_labelled_data.py`; five conditions, and a row is kept only if it satisfies all
of them:

- `retention_time` ≤ 10800 seconds
- `lower_offset` ≤ 300 Da
- `precursor_charge` ≥ 0 **and** ≤ 7 — note **0 is kept**, not excluded
- `precursor_mz` ≤ 2000 Da
- `sequence` contains no `[IN:<digits>]` token — a private namespace for modifications
  the pipeline could not resolve to a UNIMOD identifier. Measured across the corpus
  these are 175 distinct tokens, ids 3000–3174, predominantly N-glycan compositions on
  asparagine; `N[IN:3173]` and `K[IN:3174]` fall outside the internal glyco mapping,
  and the latter is on lysine, so this is **not** a glyco-only condition.
  Glycopeptides whose glycan *does* have a UNIMOD identifier are retained.

A null in any of the four numeric columns **passes** the corresponding condition
(`_nullable_filter`), so a reimplementation that drops nulls produces a different
dataset. Measured on the published corpus there are no nulls in these columns, so this
does not bite today, but it is what the code does.

Measured effect on LCFM (184,607,213 → 181,777,591 PSMs, 98.5% retained): retention
time removes ~2.54 M, the glycan condition ~385 k, and `lower_offset` 8,147. **The
charge and `precursor_mz` bounds remove nothing** — no PSM in the corpus exceeds
either, even before filtering.

**Data Validation:**
- Peptide sequences must be valid amino acid strings
- m/z and intensity arrays must have matching lengths
- All numeric values must be finite (not NaN or inf)
- Charge states must be positive integers

**Special Handling:**
- I/L ambiguity is normalized (I → L) for consistent grouping
- "Unknown" values in `collision_energy` are converted to null
- Missing `isolation_target` values are inferred from `precursor_mz`

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce chunk size or process count
- Lower safety factor
- Use memory forecasting mode

**File Not Found:**
- Check input directory paths
- Verify file permissions
- Ensure required columns exist

**Slow Performance:**
- Increase process count (if memory allows)
- Use SSD storage
- Optimize chunk sizes

### Getting Help
```bash
# Show script help
python script.py --help

# Show command help
python script.py command --help

# Verbose logging
python script.py command --verbose
```

## Advanced Configuration

### Custom Filtering
Modify filtering criteria in script parameters:
- Retention time limits
- Charge state ranges
- m/z thresholds
- Offset constraints

### Shuffling Strategies
- **Labelled data**: Peptide-level preservation
- **Unlabelled data**: LSH-based clustering
- **Global shuffling**: 2-pass algorithm

### Checkpointing
- Available for long-running processes
- Configurable checkpoint intervals
- Clear checkpoint option for fresh starts

## Reference

### Command Options Summary

| Script | Main Command | Key Options |
|--------|-------------|-------------|
| `split_labelled_data.py` | `split` | `--input-dir`, `--output-dir`, `--rows-per-file` |
| `create_labelled_subset_splits.py` | `subset` | `--input-dir`, `--split-assignments` |
| `create_consolidated_split_assignments.py` | `consolidate` | `--output-file`, `--blacklist-file` |
| `split_unlabelled_data.py` | `--mode full` | `--input-dir`, `--output-dir`, `--mz-max` |
| `shuffle_2pass.py` | `--base-dir` | `--target-chunk-size`, `--seed` |

### Batch Processing
All scripts support batch processing for multiple datasets:
```bash
python script.py batch dir1 dir2 dir3 --output-base /path/to/output
```

### Legacy Compatibility
Original functions preserved for backward compatibility:
```bash
python script.py  # Runs original main() function
```
