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

Shared flags are documented in [`../README.md`](../README.md). Every script is a single Typer command: `python <script>.py --help`. Repeat `--input-dir` / `--input-file` instead of separate batch subcommands.

### 1. Detect duplicates (`detect_all_duplicates.py`)

```bash
python detect_all_duplicates.py --input-dir ./data --output-file duplicates.txt
python detect_all_duplicates.py --input-dir ./data1 --input-dir ./data2 --output-file duplicates.txt --verbose
python detect_all_duplicates.py --input-dir ./data --extensions .ipc .parquet --output-file duplicates.txt
```

### 2. Delete same-folder duplicates (`delete_same_folder_duplicates.py`)

```bash
python delete_same_folder_duplicates.py --input-file duplicates.txt
python delete_same_folder_duplicates.py --input-file duplicates.txt --force
python delete_same_folder_duplicates.py --input-file duplicates.txt --dry-run
python delete_same_folder_duplicates.py --input-file report_a.txt --input-file report_b.txt --force
```

### 3. Find empty files (`find_empty_files.py`)

```bash
python find_empty_files.py --input-dir ./data --output-file empty_files.txt
python find_empty_files.py --input-dir ./data --min-size 1024 --output-file small_files.txt
python find_empty_files.py --input-dir ./acfm --input-dir ./lcfm --output-file empty_files.txt
```

### 4. Convert IPC to Parquet (`convert_ipc_to_parquet.py`)

```bash
python convert_ipc_to_parquet.py --input-dir ./data --output-file errors.txt
python convert_ipc_to_parquet.py --input-file file_list.txt --output-file errors.txt
python convert_ipc_to_parquet.py --input-dir ./data \
    --column-mapping '{"rt": "retention_time", "mz": "mz_array"}' --verbose
python convert_ipc_to_parquet.py --input-file list1.txt --input-file list2.txt --output-file errors.txt
```

### 5. Enforce null values (`enforce_nulls.py`)

By default replaces `"Unknown"` with null in `collision_energy` and `frag_type`.

```bash
python enforce_nulls.py --input-dir ./data --output-file affected_files.csv
python enforce_nulls.py --input-dir ./data --column some_column --old-value "N/A"
python enforce_nulls.py --input-dir ./acfm --input-dir ./lcfm --output-file affected_files.csv
```

### 6. Delete files from a list (`delete_files.py`)

```bash
python delete_files.py --input-file file_list.txt --error-log errors.txt
python delete_files.py --input-file file_list.txt --dry-run
python delete_files.py --input-file list1.txt --input-file list2.txt --error-log errors.txt
```

### 7. Find modifications (`find_modifications.py`)

```bash
python find_modifications.py --input-dir ./data --output-file modifications.xlsx
python find_modifications.py --input-dir ./data --pattern "**/*.parquet" --verbose
python find_modifications.py --input-dir ./data1 --input-dir ./data2 --output-file modifications.xlsx
```

### 8. Check conversion completeness (`check_conversion.py`)

```bash
python check_conversion.py --input-dir ./data --output-file missing_files.txt
python check_conversion.py --input-dir ./data1 --input-dir ./data2 --output-file missing_files.txt
```

### 9. Detect multi-folder duplicates (`detect_multi_folder_duplicates.py`)

```bash
python detect_multi_folder_duplicates.py --input-file duplicates.txt --output-file multi_duplicates.txt
python detect_multi_folder_duplicates.py --input-file report_a.txt --input-file report_b.txt --output-file multi_duplicates.txt
```

### 10. Delete multi-folder duplicates (`delete_multi_folder_duplicates.py`)

```bash
python delete_multi_folder_duplicates.py --input-file multi_duplicates.txt --target-dir ./target_folder
python delete_multi_folder_duplicates.py --input-file multi_duplicates.txt --target-dir ./target_folder --force
python delete_multi_folder_duplicates.py --input-file multi_duplicates.txt --target-dir ./target_folder --dry-run
```

### 11. Infer isolation targets (`infer_isolation_target.py`)

`--input-dir` accepts a glob pattern selecting parquet files.

```bash
python infer_isolation_target.py --input-dir "./data/**/*.parquet" --output-file modified.txt
python infer_isolation_target.py --input-dir "./data1/**/*.parquet" --input-dir "./data2/**/*.parquet" \
    --output-file modified.txt --error-log errors.txt
```

### 12. Label modifications (`label_modifications.py`)

```bash
python label_modifications.py --input-dir subfolder_name \
    --gold-standard-mods gold.xlsx --ambiguous-mods pxd009449.xlsx
python label_modifications.py --input-dir subfolder1 --input-dir subfolder2 \
    --gold-standard-mods gold.xlsx --ambiguous-mods pxd009449.xlsx
```

`check_modifications.py` compares inventories against the same mapping tables:

```bash
python check_modifications.py --input-file modifications.xlsx \
    --gold-standard-mods gold.xlsx --ambiguous-mods pxd009449.xlsx
```

### 13. Add acquisition column (`add_acquisition_column.py`)

Adds an `acquisition` column (DIA/DDA) from search-data Excel.

```bash
python add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data.xlsx

python add_acquisition_column.py \
    --input-dir s3://bucket/acfm/ \
    --search-data search_data.xlsx \
    --aws-profile <your-aws-profile>

python add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data.xlsx \
    --dry-run --verbose
```

**Key features:**

- Reads acquisition type from the search data Excel file (`project`, `file path`, `acquisition`)
- Skips files that already have an `acquisition` column
- Supports local directories and S3 buckets
- Dry-run mode for previewing changes

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
