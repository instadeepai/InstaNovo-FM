# Preprocessing scripts

Convert and clean mass-spectrometry files before verification and splitting.
For the full pipeline order, see [`../README.md`](../README.md).

All scripts in this folder use Typer.
Run them from the repository root as `uv run python -m scripts.preprocessing.<script>` (see [`../README.md`](../README.md)).

## Basic workflow

1. Detect duplicates, both same-folder and multi-folder. Multi-folder hits need a person to decide which copy to keep; the scripts cannot choose the true file for you.
2. Delete same-folder duplicates, then delete the reviewed multi-folder duplicates.
3. Find and remove empty or undersized files.
4. Convert IPC to Parquet, then check that every IPC file has a converted counterpart.
5. Clean columns: replace `"Unknown"` with null in `collision_energy` and `frag_type`, infer missing isolation targets, re-label EncyclopeDIA modifications as UNIMOD, and add the `acquisition` column (DIA/DDA) from search metadata.

Modification processing uses a hardcoded EncyclopeDIA → UNIMOD mapping dictionary.
For other datasets that dictionary may need updating by hand.

## Individual script usage

Shared flags are documented in [`../README.md`](../README.md). Every script is a single Typer command: `uv run python -m scripts.preprocessing.<script> --help`. Repeat `--input-dir` / `--input-file` instead of separate batch subcommands.

### 1. Detect duplicates (`detect_all_duplicates.py`)

```bash
uv run python -m scripts.preprocessing.detect_all_duplicates --input-dir ./data --output-file duplicates.txt
uv run python -m scripts.preprocessing.detect_all_duplicates --input-dir ./data1 --input-dir ./data2 --output-file duplicates.txt --verbose
uv run python -m scripts.preprocessing.detect_all_duplicates --input-dir ./data --extensions .ipc .parquet --output-file duplicates.txt
```

### 2. Delete same-folder duplicates (`delete_same_folder_duplicates.py`)

```bash
uv run python -m scripts.preprocessing.delete_same_folder_duplicates --input-file duplicates.txt
uv run python -m scripts.preprocessing.delete_same_folder_duplicates --input-file duplicates.txt --force
uv run python -m scripts.preprocessing.delete_same_folder_duplicates --input-file duplicates.txt --dry-run
uv run python -m scripts.preprocessing.delete_same_folder_duplicates --input-file report_a.txt --input-file report_b.txt --force
```

### 3. Find empty files (`find_empty_files.py`)

```bash
uv run python -m scripts.preprocessing.find_empty_files --input-dir ./data --output-file empty_files.txt
uv run python -m scripts.preprocessing.find_empty_files --input-dir ./data --min-size 1024 --output-file small_files.txt
uv run python -m scripts.preprocessing.find_empty_files --input-dir ./acfm --input-dir ./lcfm --output-file empty_files.txt
```

### 4. Convert IPC to Parquet (`convert_ipc_to_parquet.py`)

```bash
uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-dir ./data --output-file errors.txt
uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-file file_list.txt --output-file errors.txt
uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-dir ./data \
    --column-mapping '{"rt": "retention_time", "mz": "mz_array"}' --verbose
uv run python -m scripts.preprocessing.convert_ipc_to_parquet --input-file list1.txt --input-file list2.txt --output-file errors.txt
```

### 5. Enforce null values (`enforce_nulls.py`)

By default replaces `"Unknown"` with null in `collision_energy` and `frag_type`.

```bash
uv run python -m scripts.preprocessing.enforce_nulls --input-dir ./data --output-file affected_files.csv
uv run python -m scripts.preprocessing.enforce_nulls --input-dir ./data --column some_column --old-value "N/A"
uv run python -m scripts.preprocessing.enforce_nulls --input-dir ./acfm --input-dir ./lcfm --output-file affected_files.csv
```

### 6. Delete files from a list (`delete_files.py`)

```bash
uv run python -m scripts.preprocessing.delete_files --input-file file_list.txt --error-log errors.txt
uv run python -m scripts.preprocessing.delete_files --input-file file_list.txt --dry-run
uv run python -m scripts.preprocessing.delete_files --input-file list1.txt --input-file list2.txt --error-log errors.txt
```

### 7. Find modifications (`find_modifications.py`)

```bash
uv run python -m scripts.preprocessing.find_modifications --input-dir ./data --output-file modifications.xlsx
uv run python -m scripts.preprocessing.find_modifications --input-dir ./data --pattern "**/*.parquet" --verbose
uv run python -m scripts.preprocessing.find_modifications --input-dir ./data1 --input-dir ./data2 --output-file modifications.xlsx
```

### 8. Check conversion completeness (`check_conversion.py`)

```bash
uv run python -m scripts.preprocessing.check_conversion --input-dir ./data --output-file missing_files.txt
uv run python -m scripts.preprocessing.check_conversion --input-dir ./data1 --input-dir ./data2 --output-file missing_files.txt
```

### 9. Detect multi-folder duplicates (`detect_multi_folder_duplicates.py`)

```bash
uv run python -m scripts.preprocessing.detect_multi_folder_duplicates --input-file duplicates.txt --output-file multi_duplicates.txt
uv run python -m scripts.preprocessing.detect_multi_folder_duplicates --input-file report_a.txt --input-file report_b.txt --output-file multi_duplicates.txt
```

### 10. Delete multi-folder duplicates (`delete_multi_folder_duplicates.py`)

```bash
uv run python -m scripts.preprocessing.delete_multi_folder_duplicates --input-file multi_duplicates.txt --target-dir ./target_folder
uv run python -m scripts.preprocessing.delete_multi_folder_duplicates --input-file multi_duplicates.txt --target-dir ./target_folder --force
uv run python -m scripts.preprocessing.delete_multi_folder_duplicates --input-file multi_duplicates.txt --target-dir ./target_folder --dry-run
```

### 11. Infer isolation targets (`infer_isolation_target.py`)

`--input-dir` accepts a glob pattern selecting parquet files.

```bash
uv run python -m scripts.preprocessing.infer_isolation_target --input-dir "./data/**/*.parquet" --output-file modified.txt
uv run python -m scripts.preprocessing.infer_isolation_target --input-dir "./data1/**/*.parquet" --input-dir "./data2/**/*.parquet" \
    --output-file modified.txt --error-log errors.txt
```

### 12. Label modifications (`label_modifications.py`)

```bash
uv run python -m scripts.preprocessing.label_modifications --input-dir subfolder_name
uv run python -m scripts.preprocessing.label_modifications --input-dir subfolder1 --input-dir subfolder2 \
    --gold-standard-mods assets/mod_dicts/gold_standard_modifications.xlsx \
    --ambiguous-mods assets/mod_dicts/PXD009449_ambiguous_mods.xlsx
```

`check_modifications.py` compares inventories against the same mapping tables:

```bash
uv run python -m scripts.preprocessing.check_modifications --input-file modifications.xlsx
uv run python -m scripts.preprocessing.check_modifications --input-file modifications.xlsx \
    --gold-standard-mods assets/mod_dicts/gold_standard_modifications.xlsx \
    --ambiguous-mods assets/mod_dicts/PXD009449_ambiguous_mods.xlsx
```

### 13. Add acquisition column (`add_acquisition_column.py`)

Adds an `acquisition` column (DIA/DDA) from search-data Excel.
Defaults to `data/search_data.xlsx`.

```bash
uv run python -m scripts.preprocessing.add_acquisition_column \
    --input-dir <data-root>/lcfm/

uv run python -m scripts.preprocessing.add_acquisition_column \
    --input-dir s3://bucket/acfm/ \
    --search-data data/search_data.xlsx \
    --aws-profile <your-aws-profile>

uv run python -m scripts.preprocessing.add_acquisition_column \
    --input-dir <data-root>/lcfm/ \
    --search-data data/search_data.xlsx \
    --dry-run --verbose
```

**Key features:**

- Reads acquisition type from the search data Excel file (`project`, raw-filename `file path`, `acquisition`)
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
