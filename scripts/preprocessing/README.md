# Data Conversion Workflow Guide

This guide helps you process and convert mass spectrometry data using a comprehensive set of scripts. All scripts use **Typer CLI** for easy command-line interaction.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Workflow

1. **Detect and handle duplicates** (same-folder and multi-folder)
2. **Clean empty files** from datasets
3. **Convert IPC to Parquet** format
4. **Verify conversion completeness**
5. **Process data quality** (nulls, isolation targets, modifications)
6. **Add USI identifiers** for stable spectrum references (labelled parquets)
7. **Build MCFM/HCFM subsets** from scored PSM tables before train/test splitting

### Example: Complete Workflow

```bash
# 1. Detect duplicates
python detect_all_duplicates.py detect-duplicates ./data --output duplicates.txt

# 2. Handle multi-folder duplicates manually, then delete same-folder duplicates
python delete_same_folder_duplicates.py delete-duplicates duplicates.txt --force

# 3. Clean empty files
python find_empty_files.py find-empty ./data --output empty_files.txt
python delete_files.py delete empty_files.txt

# 4. Convert IPC to Parquet
python convert_ipc_to_parquet.py convert --source-dir ./data --output errors.txt

# 5. Verify conversion
python check_conversion.py check-conversion ./data --output missing_files.txt

# 6. Process data quality
python enforce_nulls.py enforce ./data --output affected_files.csv
python infer_isolation_target.py infer-targets "./data/**/*.parquet" --log modified.txt
python find_modifications.py find-mods ./data --output modifications.parquet
python label_modifications.py label-mods subfolder_name
python add_acquisition_column.py --input-dir ./data --search-data search_data.xlsx

# 7. Add PSI USI column (paths should contain a PXD accession, e.g. .../PXD009449/...)
python scripts/verification/add_usi_column.py --input-dir ./data

# 8. Medium- and high-confidence subsets (see "Create MCFM and HCFM subsets" below)
python scripts/splitting/create_subsets.py --input-dir ./lcfm_scored --medium-output-dir ./mcfm --high-output-dir ./hcfm
```

## Individual Script Usage

### 1. Detect Duplicates (`detect_all_duplicates.py`)

Finds duplicate files based on base filenames across directories.

```bash
# Detect duplicates in single directory
python detect_all_duplicates.py detect-duplicates ./data --output duplicates.txt

# Detect duplicates in multiple directories
python detect_all_duplicates.py batch-detect ./data1 ./data2 ./data3 --output-dir results

# Custom file extensions
python detect_all_duplicates.py detect-duplicates ./data --extensions .ipc .parquet --verbose
```

**Key Features:**

- Detects same-folder and multi-folder duplicates
- Supports multiple file extensions
- Batch processing for multiple directories
- Generates detailed duplicate reports

### 2. Delete Duplicates (`delete_same_folder_duplicates.py`)

Removes duplicate files from the same folder.

```bash
# Delete with confirmation
python delete_same_folder_duplicates.py delete-duplicates duplicates.txt

# Force delete without confirmation
python delete_same_folder_duplicates.py delete-duplicates duplicates.txt --force

# Dry run to preview changes
python delete_same_folder_duplicates.py delete-duplicates duplicates.txt --dry-run

# Batch delete from multiple files
python delete_same_folder_duplicates.py batch-delete duplicates1.txt duplicates2.txt --force
```

**Key Features:**

- Safe deletion with confirmation prompts
- Dry-run mode for previewing changes
- Batch processing for multiple duplicate files
- Force mode for automated workflows

### 3. Find Empty Files (`find_empty_files.py`)

Identifies empty or small files that should be removed.

```bash
# Find empty files
python find_empty_files.py find-empty ./data --output empty_files.txt

# Find files smaller than threshold
python find_empty_files.py find-empty ./data --min-size 1024 --output small_files.txt

# Batch find in multiple directories
python find_empty_files.py batch-find-empty ./acfm ./lcfm ./mcfm --output-dir results
```

**Key Features:**

- Configurable minimum file size threshold
- Custom file pattern matching
- Batch processing for multiple directories
- Detailed reporting of found files

### 4. Convert IPC to Parquet (`convert_ipc_to_parquet.py`)

Converts IPC files to Parquet format for better compatibility.

```bash
# Convert from directory
python convert_ipc_to_parquet.py convert --source-dir ./data --output errors.txt

# Convert from file list
python convert_ipc_to_parquet.py convert --input-file file_list.txt --output errors.txt

# Custom column renaming
python convert_ipc_to_parquet.py convert --source-dir ./data \
    --column-mapping '{"rt": "retention_time", "mz": "mz_array"}' \
    --verbose

# Batch convert multiple file lists
python convert_ipc_to_parquet.py batch-convert list1.txt list2.txt --output batch_errors.txt
```

**Key Features:**

- Directory-based or file-list-based conversion
- Custom column mapping support
- Error logging and reporting
- Batch processing for multiple file lists
- Lazy loading option for memory efficiency

### 5. Enforce Null Values (`enforce_nulls.py`)

Replaces specific values with null in parquet files. By default, processes the `collision_energy` and `frag_type` columns, replacing "Unknown" string values with null.

```bash
# Replace "Unknown" with null in default columns (collision_energy, frag_type)
python enforce_nulls.py enforce ./data --output affected_files.csv

# Custom column and values
python enforce_nulls.py enforce ./data --column some_column --old-value "N/A" --new-value null

# Batch process multiple directories
python enforce_nulls.py batch-enforce ./acfm ./lcfm ./mcfm --output-dir results
```

**Key Features:**

- Default columns: `collision_energy`, `frag_type`
- Replaces "Unknown" string values with proper null
- Configurable column and value replacement
- Batch processing for multiple directories
- Detailed reporting of affected files

### 6. Delete Files from List (`delete_files.py`)

Removes files listed in a text file.

```bash
# Delete files from list
python delete_files.py delete file_list.txt --error-log errors.txt

# Dry run to preview
python delete_files.py delete file_list.txt --dry-run

# Batch delete from multiple lists
python delete_files.py batch-delete list1.txt list2.txt --error-log batch_errors.txt
```

**Key Features:**

- Safe deletion with error logging
- Dry-run mode for previewing changes
- Batch processing for multiple file lists
- Comprehensive error reporting

### 7. Find Modifications (`find_modifications.py`)

Discovers modifications in parquet files.

```bash
# Find modifications
python find_modifications.py find-mods ./data --output modifications.parquet

# Custom file pattern
python find_modifications.py find-mods ./data --pattern "**/*.parquet" --verbose

# Batch find in multiple directories
python find_modifications.py batch-find-mods ./data1 ./data2 --output-dir results
```

**Key Features:**

- Detects peptide modifications in mass spec data
- Custom file pattern matching
- Batch processing for multiple directories
- Outputs to Excel format for analysis

### 8. Check Conversion Completeness (`check_conversion.py`)

Verifies IPC to Parquet conversion was complete.

```bash
# Check conversion completeness
python check_conversion.py check-conversion ./data --output missing_files.txt

# Batch check multiple directories
python check_conversion.py batch-check ./data1 ./data2 --output-dir results
```

**Key Features:**

- Identifies missing converted files
- Batch processing for multiple directories
- Detailed reporting of conversion gaps
- Essential for data integrity verification

### 9. Detect Multi-Folder Duplicates (`detect_multi_folder_duplicates.py`)

Finds duplicates across different folders.

```bash
# Detect multi-folder duplicates
python detect_multi_folder_duplicates.py detect-duplicates duplicates.txt --output multi_duplicates.txt

# Batch detect from multiple files
python detect_multi_folder_duplicates.py batch-detect duplicates1.txt duplicates2.txt --output-dir results
```

**Key Features:**

- Identifies duplicates across different directories
- Requires manual resolution (user input needed)
- Batch processing for multiple duplicate files
- Detailed cross-folder duplicate reporting

### 10. Delete Multi-Folder Duplicates (`delete_multi_folder_duplicates.py`)

Removes multi-folder duplicates after manual review.

```bash
# Delete with confirmation
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder

# Force delete without confirmation
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder --force

# Dry run to preview
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder --dry-run
```

**Key Features:**

- Safe deletion with confirmation prompts
- Dry-run mode for previewing changes
- Batch processing for multiple files
- Target folder specification

### 11. Infer Isolation Targets (`infer_isolation_target.py`)

Infers missing isolation target values in parquet files.

```bash
# Infer isolation targets
python infer_isolation_target.py infer-targets "./data/**/*.parquet" --log modified.txt

# Batch infer in multiple directories
python infer_isolation_target.py batch-infer "./data1/**/*.parquet" "./data2/**/*.parquet" --log-dir results
```

**Key Features:**

- Infers missing isolation target values from experiment header metadata
- Uses precursor m/z as fallback
- Batch processing for multiple patterns
- Detailed logging of modified files

### 12. Label Modifications (`label_modifications.py`)

Converts modifications to UNIMOD format.

```bash
# Label modifications in subfolder
python label_modifications.py label-mods subfolder_name

# Batch label in multiple subfolders
python label_modifications.py batch-label-mods subfolder1 subfolder2 subfolder3
```

**Key Features:**

- Converts EncyclopeDIA modifications to UNIMOD format
- Hardcoded modification mapping dictionary
- Batch processing for multiple subfolders
- Essential for standardisation

### 13. Add Acquisition Column (`add_acquisition_column.py`)

Adds an `acquisition` column to parquet files based on search data. The column value is either "DIA" or "DDA" as specified in the search data Excel file.

```bash
# Add acquisition column from search data
python add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data_with_new_projects.xlsx

# S3 bucket support
python add_acquisition_column.py \
    --input-dir s3://bucket/acfm/ \
    --search-data search_data.xlsx \
    --aws-profile <your-aws-profile>

# Dry run to preview changes
python add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data.xlsx \
    --dry-run --verbose
```

**Key Features:**

- Reads acquisition type from search data Excel file (columns: `project`, `file path`, `acquisition`)
- Adds `acquisition` column with value "DIA" or "DDA" to each parquet file
- Skips files that already have an `acquisition` column
- Supports both local directories and S3 buckets
- Dry-run mode for previewing changes
- Reports files not found in search data

## Data Verification Scripts

These scripts verify data integrity and identify issues that need correction before training.

### Add USI column (`../verification/add_usi_column.py`)

Adds a `usi` column to labelled parquet files using the [PSI USI](https://www.psidev.info/usi) format via `pyteomics.usi.USI`. Each row gets an `mzspec:<PXD>:<datafile>:<scanType>:<scan>:<interpretation>` string when the required fields are present.

**Inputs used per row:**

- **PXD accession**: First `PXD######` or `MSV######` token found in the file path (project folders should include the accession).
- **Data file**: Basename derived from the parquet path (same stem logic as other pipeline scripts).
- **Scan**: Normalized from the `scan` column (plain integers or vendor text such as `scan=1321`).
- **Interpretation**: `sequence` and `precursor_charge` as `sequence/z` when charge is an integer.

**Requirements:**

- Parquet must include `scan`; labelled rows should include `sequence` and `precursor_charge` for a full interpretation segment.
- Files whose basename contains `:` are skipped (USI encoding limitation).

```bash
# All project subfolders under the root
python scripts/verification/add_usi_column.py --input-dir <data-root>/lcfm/
```

*Notes:*

- Sequences with internal EncyclopeDIA-style tokens such as `[IN:…]` are passed through as-is in the interpretation; they are not strict ProForma and may not resolve in public PROXI services until converted to UNIMOD-style notation.
- Each labelled row is assigned the `<interpretation>` `<sequence>/<precursor_charge>`. DIA sequences are assigned interpretations with zero precursor charges.
- Projects with MassIVE project identifiers instead of PRIDE identifiers are passed through as-is, but generally PXD identifiers are preferred to MSV.

### Verify Intensity Arrays are Max-Normalised (`../verification/verify_intensity_max_normalisation.py`)

Verifies that all spectra have intensities normalised by their maximum value, storing this max in the `scale_factor` column.

```bash
python scripts/verification/verify_intensity_max_normalisation.py --input-dir <data-root>/lcfm
```

```bash
# Normalise rows in-place
python scripts/verification/verify_intensity_max_normalisation.py --input-dir <data-root>/lcfm --fix
```

### Verify Precursor Charges and Acquisition Type (`../verification/verify_precursor_charges_and_acq_type.py`)

Verifies that precursor charge values are consistent with the acquisition type (DDA vs DIA).

- **DDA files** should have non-zero precursor charges (charge state is known)
- **DIA files** should have zero precursor charges (charge state is unknown due to wide isolation windows)

```bash
# Verify precursor charges against acquisition type
python scripts/verification/verify_precursor_charges_and_acq_type.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data_with_new_projects.xlsx \
    --output-dir lcfm

# S3 bucket support
python scripts/verification/verify_precursor_charges_and_acq_type.py \
    --input-dir s3://bucket/acfm/ \
    --search-data search_data.xlsx \
    --output-dir acfm \
    --aws-profile <your-aws-profile>
```

**Output Files:**

- `incorrect_dia_files.csv`: DIA files with non-zero precursor charges
- `incorrect_dda_files.csv`: DDA files with zero/null/unknown precursor charges
- `project_summary.csv`: Project-level summary of errors

**Handling Discrepancies:**

- **Problematic DIA files**: Remove entire files (non-zero charges indicate mislabeled acquisition type)
- **Problematic DDA rows**: Remove individual rows with zero/null charges (partial data quality issue)

### Verify Calculated m/z (`../verification/verify_calc_mz.py`)

Verifies whether the `peptide_calc_mz` column matches our calculation from the `sequence` column. This helps identify:

1. **Implicit cysteine carbamidomethylation**: Sequences with cysteines written as "C" but calculated with C[UNIMOD:4] mass
2. **Implicit TMT and iTRAQ tagging**: Sequences with lysines written as "K" but calculated with K[tag] mass (i.e. iTRAQ4, TMT10, TMT18 tags).
3. **EncyclopeDIA → Proforma PTM mistranslation**: Incorrect modification mapping during conversion
4. **Incorrect precursor charge**: Incorrect charges used in peptide m/z calculation.

```bash
# With search metadata, TMTplex grouping, and optional lysine-label report
python scripts/verification/verify_calc_mz.py \
    --input-dir <data-root>/lcfm/ \
    --output-csv calc_mz_verification.csv \
    --search-data search_data_with_new_projects.xlsx \
    --tmt-projects-yaml bad_tmt_projects.yaml \
    --lysine-label-file-csv lysine_label_files.csv
```

## Data Splitting Scripts

### Split Labelled Data (`../splitting/split_labelled_data.py`)

Splits labelled MS/MS spectra into train/test/validation sets with no peptide leakage.

**Key Features:**

- **No peptide leakage**: Uses a peptide registry from HuggingFace to ensure the same peptide sequence doesn't appear in multiple splits
- **Peptide normalisation**: Strips [UNIMOD:XX] modifications and applies I→L mapping for consistent lookup
- **Quality filters**: Applies retention_time, lower_offset, precursor_charge, and precursor_mz filters, and discards PSMs whose `sequence` carries a modification the pipeline could not resolve to a UNIMOD identifier (the custom `[IN:<digits>]` annotation — 175 tokens over ids 3000–3174, predominantly N-glycans on asparagine but not exclusively: `K[IN:3174]` is on lysine).
- **80/10/10 split ratio**: New peptides are assigned using this ratio

```bash
# Single directory - update registry and split files
python scripts/splitting/split_labelled_data.py split \
    --input-dir lcfm \
    --output-dir lcfm_splits

# Update registry only (no split files)
python scripts/splitting/split_labelled_data.py split \
    --input-dir lcfm \
    --output-dir lcfm_splits \
    --mode update-splits

# Split files only (registry must contain all peptides)
python scripts/splitting/split_labelled_data.py split \
    --input-dir lcfm \
    --output-dir lcfm_splits \
    --mode split-only

# Multiple directories (single combined pass)
python scripts/splitting/split_labelled_data.py batch dir1 dir2 dir3 --output-dir splits/
```

**Run Modes:**

- `update-splits`: Collect unique peptides, assign new ones 80/10/10, save registry (no split files)
- `split-only`: Split files using existing registry (errors if peptide not found)
- `both` (default): Performs both operations in sequence

**Quality Filters Applied:**

- `retention_time` <= 10800 seconds (3 hours)
- `lower_offset` <= 300 Da
- `precursor_charge` in range [0, 7]
- `precursor_mz` <= 2000 Da
- `sequence` cannot contain PTMs with the format `[IN:<digits>]`

### Create MCFM and HCFM subsets (`../splitting/create_subsets.py`)

Builds **medium-confidence (MCFM)** and **high-confidence (HCFM)** parquet trees from a root directory of all-confidence labelled spectra (one subfolder per dataset, `.parquet` files inside). This is separate from `split_labelled_data.py`: it filters rows by a **global** score threshold, not by train/test/val splits.

#### How scoring works

1. **Composite score** (temporary column `_composite_score`) combines EncyclopeDIA-style scores, each normalized as a **percentile rank within peptide length** (matching grouped `rank(pct=True, method="average")` semantics):
   - `1 - expectation`, `probability`, and `hyperscore` contribute ranked components.
   - If `nextscore` is present, the term `(hyperscore - nextscore)` is included and the average is over four parts; otherwise three parts.
   - Peptide length comes from a `peptide_length` or `peptide length` column if present, else uppercase-letter count in `peptide` or `unmodified_peptide`.

2. **Pass 1** reads every file, computes composite scores (for finite values only), and concatenates them.

3. **Global thresholds** (linear quantiles on the pooled scores):
   - **MCFM**: rows with `_composite_score` **strictly greater** than the **90th percentile** (~top 10% of PSMs globally).
   - **HCFM**: rows with `_composite_score` **strictly greater** than the **98th percentile** (~top 2% globally).

4. **Pass 2** writes filtered parquets to `--medium-output-dir` and `--high-output-dir`, preserving the same subfolder and filename layout as the input.

**Optional:** `--hold-back-modified-rows` drops rows whose `sequence` contains internal modification tokens `[IN:<digits>]` before scoring and output (so those rows do not affect thresholds or exported subsets).

```bash
uv run python scripts/splitting/create_subsets.py \
    --input-dir <data-root>/lcfm/ \
    --medium-output-dir <data-root>/mcfm_new/ \
    --high-output-dir <data-root>/hcfm_new/ \
    --hold-back-modified-rows
```

## Data Processing Workflow

### Complete Workflow Steps

1. **Detect Duplicates**: Find all same-folder and multi-folder duplicates
2. **Handle Multi-Folder Duplicates**: Manual review required for cross-folder duplicates
3. **Delete Same-Folder Duplicates**: Remove duplicates within same folder
4. **Clean Empty Files**: Remove empty or small files
5. **Convert IPC to Parquet**: Convert data format for better compatibility
6. **Verify Conversion**: Ensure all files were converted successfully
7. **Process Data Quality**:
   - Enforce null values for "Unknown" entries in `collision_energy` and `frag_type`
   - Infer missing isolation targets
   - Find and re-label modifications
   - Add acquisition column (DIA/DDA) from search data
8. **Add USI Column**: Add a universal spectrum identifier column that optionally includes sequence label information
9. **MCFM/HCFM Creation**: From all-confidence LCFM, run `scripts/splitting/create_subsets.py` to emit global top-10% and top-2% subset trees
10. **Verify Data Integrity**:
   - Verify intensity arrays are max-normalised
   - Check precursor charges match acquisition type (DDA/DIA)
   - Verify calculated m/z values for implicit carbamidomethylation or TMT/iTRAQ tagging
   - Remove problematic DIA files or DDA rows as needed
11. **Split Labelled Data**: Partition into train/test/validation with no peptide leakage
12. **Final Verification**: Double-check duplicate handling
13. **Globally shuffle dataset splits**

### Workflow Considerations

**Multi-Folder Duplicates**: These require manual resolution because the system cannot automatically determine which folder contains the "true" file. Use the detection and deletion scripts to help manage this process.

**Modification Processing**: This step converts EncyclopeDIA modifications to standard UNIMOD format using a hardcoded modification mapping dictionary. For other datasets, this may need to be manually updated.

### Getting Help

```bash
# Show script help
python script.py --help

# Show command help
python script.py command --help

# Verbose logging
python script.py command --verbose
```

### Legacy Compatibility

Original functions preserved for backward compatibility:

```bash
python script.py  # Runs original main() function
```

## Data Format Requirements

### Required Columns and Datatypes

#### For IPC Conversion (Input IPC Files)

| Column | Datatype | Description | Required |
| -------- | ---------- | ------------- | ---------- |
| `index` | `int` | Row index | Yes |
| `scan` | `int` | Scan number | Yes |
| `header` | `str` | Experiment header | Yes |
| `rt` | `float` | Retention time in seconds | Yes |
| `frag_type` | `str` | Fragmentation type | Yes |
| `collision_energy` | `str` | Collision energy (can be "Unknown") | Yes |
| `precursor_mz` | `float` | Precursor m/z value | Yes |
| `isolation_target` | `float` | Isolation target m/z (can be null) | Yes |
| `mz` | `list[float]` | m/z array for spectrum | Yes |
| `intensity` | `list[float]` | Intensity array for spectrum | Yes |
| `scale_factor` | `float` | Intensity scaling factor | Yes |
| `peptide` | `str` | Peptide sequence | Yes |

### Data Quality Requirements

**Conversion Validation:**

- IPC files must be valid Arrow format
- All required columns must be present
- m/z and intensity arrays must have matching lengths
- All numeric values must be finite (not NaN or inf)

**Processing Requirements:**

- Peptide sequences must be valid amino acid strings
- File paths must be accessible and readable
- Directory structures must be consistent
- Modifications must be in EncyclopeDIA format

**Special Handling:**

- "Unknown" values in `collision_energy` and `frag_type` are converted to null
- Missing `isolation_target` values are inferred from `precursor_mz`
- Modifications are converted from EncyclopeDIA to UNIMOD format
- File names are preserved during conversion (.ipc → .parquet)
- Implicit cysteine carbamidomethylation and lysine tagging is detected via m/z verification
- DDA files with zero precursor charges are removed
- DIA files with non-zero precursor charges are removed
- Before subset creation and splitting, spectral quality filters are applied to remove abnormal or non-Proforma-compliant samples

### Column Mapping for IPC Conversion

Default column mapping (can be customized):

```json
{
  "rt": "retention_time",
  "mz": "mz_array",
  "intensity": "intensity_array"
}
```
