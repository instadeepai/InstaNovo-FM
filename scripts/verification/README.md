# Verification scripts

Check labelled parquet trees after preprocessing and before splitting.
Several of these scripts can also apply the corresponding repair.
For the full pipeline order, see [`../README.md`](../README.md).

Run these from the repository root as `uv run python -m scripts.verification.<script>` (see [`../README.md`](../README.md)).

## Basic workflow

1. Add USI identifiers on labelled parquets. Paths should contain a PXD or MSV accession.
2. Verify data integrity, then apply the matching repair scripts as needed:
   - intensity arrays max-normalised, with the maximum kept in `scale_factor`
   - precursor charges consistent with acquisition type; drop mislabelled DIA files and bad DDA rows
   - calculated m/z against the sequence, catching implicit carbamidomethylation, TMT/iTRAQ tagging, EncyclopeDIA mistranslation and wrong charges
3. Build or refresh the UNIMOD residue-mass dictionary when the modification tables change, then re-run calculated m/z checks.

## Individual script usage

### 1. Add USI column (`add_usi_column.py`)

Adds a `usi` column to parquet files in the [PSI USI](https://www.psidev.info/usi) format, built with `pyteomics.usi.USI`.
The string is `mzspec:<collection>:<datafile>:<scanType>:<scan>[:<interpretation>]` when the locator fields can be filled.

```bash
uv run python -m scripts.verification.add_usi_column --input-dir <data-root>/lcfm/
```

#### Inputs used per row

- Collection: the first `PXD######` or `MSV######` token in the file path. Project folders should include that accession. MassIVE identifiers are accepted, but PXD is preferred.
- Data file: the experiment stem from the parquet path, after stripping shard suffixes and embedded `.mzml`, the same key as `search_data_lookup_key`. If the row has a `filepath` column, that path is used instead of the parquet path.
- Scan: a numeric scan identifier from the `scan` column, either a plain integer or vendor text such as `scan=1321`.
- Interpretation, when `sequence` is present:
  - DDA (integer charge other than 0): `<sequence>/<charge>`
  - DIA (charge missing or 0): `<sequence>` only, with no `/0` suffix
  - Internal `[IN:<digits>]` tokens are copied through as-is. They are not valid ProForma, so those USIs will not resolve against public spectrum repositories.

#### When a row gets a null USI

The parquet is still written, with `usi` null for that row, if a locator field cannot be filled:

- no `PXD######` or `MSV######` in the path
- empty experiment stem, or `search_data_lookup_key` raising
- `scan` missing or not parseable as a number
- a `:` in the experiment stem (USI fields are colon-delimited). The script logs a warning for that stem and continues.

### 2. Verify intensity arrays are max-normalised (`verify_intensity_max_normalisation.py`)

Verifies that every spectrum has intensities normalised by their maximum value, with that maximum kept in the `scale_factor` column.

```bash
uv run python -m scripts.verification.verify_intensity_max_normalisation --input-dir <data-root>/lcfm
```

Add `--fix` to normalise non-conforming rows in place:

```bash
uv run python -m scripts.verification.verify_intensity_max_normalisation --input-dir <data-root>/lcfm --fix
```

### 3. Verify precursor charges and acquisition type (`verify_precursor_charges_and_acq_type.py`)

Verifies that precursor charge values are consistent with the acquisition type.

- DDA files should have non-zero precursor charges, because the charge state is known.
- DIA files should have zero precursor charges, because the wide isolation windows leave the charge state unknown.

```bash
uv run python -m scripts.verification.verify_precursor_charges_and_acq_type \
    --input-dir <data-root>/lcfm/ \
    --output-dir lcfm

uv run python -m scripts.verification.verify_precursor_charges_and_acq_type \
    --input-dir s3://bucket/acfm/ \
    --search-data data/search_data.xlsx \
    --output-dir acfm \
    --aws-profile <your-aws-profile>
```

#### Output files

- `incorrect_dia_files.csv` — DIA files with non-zero precursor charges.
- `incorrect_dda_files.csv` — DDA files with zero, null or unknown precursor charges.
- `project_summary.csv` — project-level summary of errors.

#### Handling discrepancies

- Problematic DIA files: remove the entire file, since non-zero charges indicate a mislabelled acquisition type.
- Problematic DDA rows: remove the individual rows with zero or null charges, since this is a partial data quality issue.

`fix_precursor_charges_from_reports.py` applies both rules from those CSV reports, deleting the DIA files and dropping the bad DDA rows.

### 4. Verify calculated m/z (`verify_calc_mz.py`)

Verifies whether the `peptide_calc_mz` column matches our own calculation from the `sequence` column.
This identifies four problems:

1. Implicit cysteine carbamidomethylation, where cysteines are written as `C` but the mass was calculated as `C[UNIMOD:4]`.
2. Implicit TMT and iTRAQ tagging, where lysines are written as `K` but the mass was calculated with a tag such as iTRAQ4, TMT10 or TMT18.
3. EncyclopeDIA to ProForma mistranslation, meaning an incorrect modification mapping during conversion.
4. Incorrect precursor charges used in the peptide m/z calculation.

```bash
uv run python -m scripts.verification.verify_calc_mz \
    --input-dir <data-root>/lcfm/ \
    --output-file calc_mz_verification.csv \
    --search-data data/search_data.xlsx \
    --tmt-projects-yaml assets/bad_tmt_projects.yaml \
    --lysine-label-file-csv lysine_label_files.csv
```

#### Related repair scripts

- `apply_carbamido_from_calc_mz_report.py` and `apply_carbamido_manual_projects.py` — rewrite bare `C` as `C[UNIMOD:4]` when the report, or a manual project list, shows the project is carbamidomethylated.
- `apply_tmt_itraq_from_search_data.py` — label bare `K` with the TMT or iTRAQ UNIMOD indicated by the search metadata.

### 5. Build UNIMOD mass dictionary (`build_unimod_mass_dictionary.py`)

Builds a residue-mass YAML that `verify_calc_mz.py` can use to recompute peptide m/z from `sequence`.
It does not walk parquet files, but instead starts from the same gold-standard and PXD009449-ambiguous Excel tables as `label_modifications.py`.

```bash
uv run python -m scripts.verification.build_unimod_mass_dictionary \
    --gold-standard-mods assets/mod_dicts/gold_standard_modifications.xlsx \
    --ambiguous-mods assets/mod_dicts/PXD009449_ambiguous_mods.xlsx \
    --output-dir assets/mod_dicts
```

#### What it does

1. Concatenate the two Excel files and split rows by `proposed_unimod_encoding`. Rows containing `UNIMOD:` go into the mass dictionary. Rows containing `IN:` are custom annotations that have no UNIMOD mass, so they are written out for expert review instead of being scored.
2. Parse each EncyclopeDIA token (`K[156]`, `n[43]A`, …) to record which amino acids, or N-terminus, were actually observed with each UNIMOD ID. Only those observed combinations are kept, not every site UNIMOD lists.
3. Download and cache `unimod_tables.xml` (first run only), then look up title, monoisotopic mass and allowed sites for each id.
4. Warn when an observed amino acid is not a known UNIMOD site for that modification, which can mean a mapping error in the Excel tables.
5. Write masses for the 20 standard amino acids plus each observed token. Residue mods are `K[UNIMOD:121]` (amino-acid mass plus delta). N-terminal mods are `[UNIMOD:1]` (delta only). `J` is omitted because I and L are collapsed elsewhere.

#### Outputs (under `--output-dir`, default `assets/mod_dicts/`)

- `residue_masses.yaml` — token to monoisotopic mass, consumed by `verify_calc_mz.py --residue-masses-file`.
- `modification_validation_report.md` — human-readable list of each UNIMOD id, observed sites, masses, and any suspect site pairings.
- `custom_modifications_for_annotation.xlsx` — unique `IN:` encodings with project, an example file name, and how often they appear.

## Data requirements

Labelled parquet trees should already have been converted and column-cleaned in preprocessing.
Charge and calculated-m/z checks also need the search-data Excel used to assign acquisition type.

| Column or input | Used by | Notes |
| --- | --- | --- |
| File path containing `PXD######` or `MSV######` | USI | MassIVE identifiers are accepted; PXD is preferred |
| `scan` | USI | Integer or vendor text such as `scan=1321` |
| `sequence` | USI, calculated m/z | Optional for USI interpretation; required for m/z checks |
| `filepath` | USI | Optional override of the parquet path when present |
| `intensity` / intensity array, `scale_factor` | Intensity normalisation | Maximum intensity should be stored in `scale_factor` |
| `precursor_charge`, acquisition type | Charge checks | DDA: non-zero charges; DIA: zero charges |
| `peptide_calc_mz` | Calculated m/z | Compared with a mass recomputed from `sequence` |
| Search-data Excel | Charge checks, calculated m/z, TMT/iTRAQ repair | Must include `project`, raw-filename `file path`, and `acquisition` (defaults to `data/search_data.xlsx`) |
