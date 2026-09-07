# Preprocessing scripts

Convert and clean mass-spectrometry files before verification and splitting.
For the full pipeline order, see [`../README.md`](../README.md).

All scripts in this folder use Typer.
Run them from this directory, or give the full path from the repository root as `python scripts/preprocessing/<script>.py`.

## Basic workflow

1. Detect duplicates, both same-folder and multi-folder. Multi-folder hits need a person to decide which copy to keep; the scripts cannot choose the true file for you.
2. Delete same-folder duplicates, then delete the reviewed multi-folder duplicates.
3. Find and remove empty or undersized files.
4. Convert IPC to Parquet, then check that every IPC file has a converted counterpart.
5. Clean columns: replace `"Unknown"` with null in `collision_energy` and `frag_type`, infer missing isolation targets, re-label EncyclopeDIA modifications as UNIMOD, and add the `acquisition` column (DIA/DDA) from search metadata.

Modification processing uses a hardcoded EncyclopeDIA → UNIMOD mapping dictionary.
For other datasets that dictionary may need updating by hand.

## Individual script usage

### 1. Detect duplicates (`detect_all_duplicates.py`)

Finds duplicate files based on base filenames across directories.

```bash
# Detect duplicates in single directory
python detect_all_duplicates.py detect-duplicates ./data --output duplicates.txt

# Detect duplicates in multiple directories
python detect_all_duplicates.py batch-detect ./data1 ./data2 ./data3 --output-dir results

# Custom file extensions
python detect_all_duplicates.py detect-duplicates ./data --extensions .ipc .parquet --verbose
```

#### Key features

- Detects same-folder and multi-folder duplicates
- Supports multiple file extensions
- Batch processing for multiple directories
- Generates detailed duplicate reports

### 2. Delete duplicates (`delete_same_folder_duplicates.py`)

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

**Key features:**

- Safe deletion with confirmation prompts
- Dry-run mode for previewing changes
- Batch processing for multiple duplicate files
- Force mode for automated workflows

### 3. Find empty files (`find_empty_files.py`)

Identifies empty or small files that should be removed.

```bash
# Find empty files
python find_empty_files.py find-empty ./data --output empty_files.txt

# Find files smaller than threshold
python find_empty_files.py find-empty ./data --min-size 1024 --output small_files.txt

# Batch find in multiple directories
python find_empty_files.py batch-find-empty ./acfm ./lcfm ./mcfm --output-dir results
```

**Key features:**

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

**Key features:**

- Directory-based or file-list-based conversion
- Custom column mapping support
- Error logging and reporting
- Batch processing for multiple file lists
- Lazy loading option for memory efficiency

### 5. Enforce null values (`enforce_nulls.py`)

Replaces specific values with null in parquet files.
By default it processes the `collision_energy` and `frag_type` columns, replacing the string "Unknown" with null.

```bash
# Replace "Unknown" with null in default columns (collision_energy, frag_type)
python enforce_nulls.py enforce ./data --output affected_files.csv

# Custom column and values
python enforce_nulls.py enforce ./data --column some_column --old-value "N/A" --new-value null

# Batch process multiple directories
python enforce_nulls.py batch-enforce ./acfm ./lcfm ./mcfm --output-dir results
```

**Key features:**

- Default columns: `collision_energy` and `frag_type`
- Replaces "Unknown" string values with proper null
- Configurable column and value replacement
- Batch processing for multiple directories
- Detailed reporting of affected files

### 6. Delete files from a list (`delete_files.py`)

Removes files listed in a text file.

```bash
# Delete files from list
python delete_files.py delete file_list.txt --error-log errors.txt

# Dry run to preview
python delete_files.py delete file_list.txt --dry-run

# Batch delete from multiple lists
python delete_files.py batch-delete list1.txt list2.txt --error-log batch_errors.txt
```

**Key features:**

- Safe deletion with error logging
- Dry-run mode for previewing changes
- Batch processing for multiple file lists
- Comprehensive error reporting

### 7. Find modifications (`find_modifications.py`)

Discovers modifications in parquet files.

```bash
# Find modifications
python find_modifications.py find-mods ./data --output modifications.parquet

# Custom file pattern
python find_modifications.py find-mods ./data --pattern "**/*.parquet" --verbose

# Batch find in multiple directories
python find_modifications.py batch-find-mods ./data1 ./data2 --output-dir results
```

**Key features:**

- Detects peptide modifications in mass spec data
- Custom file pattern matching
- Batch processing for multiple directories
- Outputs to Excel format for analysis

### 8. Check conversion completeness (`check_conversion.py`)

Verifies that the IPC to Parquet conversion was complete.

```bash
# Check conversion completeness
python check_conversion.py check-conversion ./data --output missing_files.txt

# Batch check multiple directories
python check_conversion.py batch-check ./data1 ./data2 --output-dir results
```

**Key features:**

- Identifies missing converted files
- Batch processing for multiple directories
- Detailed reporting of conversion gaps
- Essential for data integrity verification

### 9. Detect multi-folder duplicates (`detect_multi_folder_duplicates.py`)

Finds duplicates across different folders.

```bash
# Detect multi-folder duplicates
python detect_multi_folder_duplicates.py detect-duplicates duplicates.txt --output multi_duplicates.txt

# Batch detect from multiple files
python detect_multi_folder_duplicates.py batch-detect duplicates1.txt duplicates2.txt --output-dir results
```

**Key features:**

- Identifies duplicates across different directories
- Requires manual resolution, so user input is needed
- Batch processing for multiple duplicate files
- Detailed cross-folder duplicate reporting

### 10. Delete multi-folder duplicates (`delete_multi_folder_duplicates.py`)

Removes multi-folder duplicates after manual review.

```bash
# Delete with confirmation
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder

# Force delete without confirmation
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder --force

# Dry run to preview
python delete_multi_folder_duplicates.py delete-duplicates multi_duplicates.txt ./target_folder --dry-run
```

**Key features:**

- Safe deletion with confirmation prompts
- Dry-run mode for previewing changes
- Batch processing for multiple files
- Target folder specification

### 11. Infer isolation targets (`infer_isolation_target.py`)

Infers missing isolation target values in parquet files.

```bash
# Infer isolation targets
python infer_isolation_target.py infer-targets "./data/**/*.parquet" --log modified.txt

# Batch infer in multiple directories
python infer_isolation_target.py batch-infer "./data1/**/*.parquet" "./data2/**/*.parquet" --log-dir results
```

**Key features:**

- Infers missing isolation target values from experiment header metadata
- Uses precursor m/z as fallback
- Batch processing for multiple patterns
- Detailed logging of modified files

### 12. Label modifications (`label_modifications.py`)

Converts modifications to UNIMOD format.

```bash
# Label modifications in subfolder
python label_modifications.py label-mods subfolder_name

# Batch label in multiple subfolders
python label_modifications.py batch-label-mods subfolder1 subfolder2 subfolder3
```

**Key features:**

- Converts EncyclopeDIA modifications to UNIMOD format
- Hardcoded modification mapping dictionary
- Batch processing for multiple subfolders
- Essential for standardisation

`check_modifications.py` compares labelled sequences against the same mapping tables.

### 13. Add acquisition column (`add_acquisition_column.py`)

Adds an `acquisition` column to parquet files based on the search data.
The value is either "DIA" or "DDA", as specified in the search data Excel file.

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

**Key features:**

- Reads acquisition type from the search data Excel file, using the `project`, `file path` and `acquisition` columns
- Adds an `acquisition` column with the value "DIA" or "DDA" to each parquet file
- Skips files that already have an `acquisition` column
- Supports both local directories and S3 buckets
- Dry-run mode for previewing changes
- Reports files not found in the search data

## Data requirements

### Required columns and datatypes for IPC conversion

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

### Data quality requirements

**Conversion validation:**

- IPC files must be valid Arrow format
- All required columns must be present
- m/z and intensity arrays must have matching lengths
- All numeric values must be finite, so no NaN or infinity

**Processing requirements:**

- Peptide sequences must be valid amino acid strings
- File paths must be accessible and readable
- Directory structures must be consistent
- Modifications must be in EncyclopeDIA format

**Special handling:**

- "Unknown" values in `collision_energy` and `frag_type` are converted to null
- Missing `isolation_target` values are inferred from `precursor_mz`
- Modifications are converted from EncyclopeDIA to UNIMOD format
- File names are preserved during conversion, with `.ipc` becoming `.parquet`

### Column mapping for IPC conversion

Default column mapping, which can be customised:

```json
{
  "rt": "retention_time",
  "mz": "mz_array",
  "intensity": "intensity_array"
}
```
