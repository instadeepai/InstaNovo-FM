# Data pipeline

Scripts in this folder turn public mass-spectrometry submissions into the InstaNovo-FM training corpora (ACFM, LCFM, MCFM, HCFM).
Run the stages in order.
Each stage has its own README covering CLI flags, scoring rules and column requirements.

| Stage | Folder | What it does |
| --- | --- | --- |
| 1. Preprocessing | [`preprocessing/`](preprocessing/README.md) | Deduplicate, convert IPC → Parquet, clean columns, standardise modifications |
| 2. Verification | [`verification/`](verification/README.md) | Check and, where needed, repair spectrum identifiers, charges, intensities and calculated m/z |
| 3. Splitting | [`splitting/`](splitting/README.md) | Build MCFM/HCFM subsets, peptide-disjoint train/test/val splits, shuffle |

## Recommended order

1. Detect and handle duplicates, both same-folder and multi-folder. Multi-folder hits need a person to decide which copy to keep.
2. Clean empty files.
3. Convert IPC to Parquet, then verify conversion completeness.
4. Process data quality:
   - replace `"Unknown"` with null in `collision_energy` and `frag_type`
   - infer missing isolation targets
   - find and re-label EncyclopeDIA modifications as UNIMOD
   - add the `acquisition` column (DIA/DDA) from search metadata
5. Add USI identifiers on labelled parquets. Paths should contain a PXD or MSV accession.
6. Verify data integrity, then apply the matching repair scripts as needed:
   - intensity arrays max-normalised, with the maximum kept in `scale_factor`
   - precursor charges consistent with acquisition type
   - calculated m/z against the sequence, catching implicit carbamidomethylation, TMT/iTRAQ tagging, EncyclopeDIA mistranslation and wrong charges
   - drop mislabelled DIA files and bad DDA rows
7. Build quality-filtered subsets MCFM and HCFM from scored LCFM with `create_subsets.py`.
8. Split labelled data (LCFM first, then MCFM and HCFM) against the peptide registry so no peptide leaks across train, test and validation. Unlabelled ACFM is split separately by LSH clustering.
9. Shuffle each split globally.

Steps 1 to 4 are documented in [`preprocessing/README.md`](preprocessing/README.md).
Steps 5 and 6 are documented in [`verification/README.md`](verification/README.md).
Steps 7 to 9 are documented in [`splitting/README.md`](splitting/README.md).

## CLI

Scripts use Typer. Most expose a single command (no subcommand), so:

```bash
python scripts/<stage>/<script>.py --help
python scripts/<stage>/<script>.py --input-dir ...
```

`split_labelled_data.py` is the exception: it keeps `split` and `batch` because `batch` is one combined registry pass, not a for-loop over independent runs.

### Shared flags

Use these names when a script needs the concept. Not every script takes every flag.

| Role | Flag | Short | Notes |
| --- | --- | --- | --- |
| Data tree | `--input-dir` | `-i` | Repeatable. |
| Path list / report in | `--input-file` | — | Repeatable. `-i` is reserved for `--input-dir`. |
| Single parquet/ipc | `--input` | — | Intensity verify only; mutually exclusive with `--input-dir`. |
| Output tree | `--output-dir` | — | Avoid `-o` so it cannot mean a file. |
| Output file / report | `--output-file` | `-o` | |
| Search Excel | `--search-data` | — | Required when the script uses it; no default filename. |
| Gold / ambiguous mods | `--gold-standard-mods`, `--ambiguous-mods` | — | Options, not positionals. |
| Project filter | `--project` | `-p` | Only use of `-p`. |
| Dry run | `--dry-run` | `-n` | |
| Verbose | `--verbose` | `-v` | |
| Force (skip confirm) | `--force` | — | |
| AWS | `--aws-profile` | — | |
| Error log | `--error-log` | — | |

Keep long, specific names with no short flag when two path-like things could be confused: `--registry-dir`, `--report-dir`, `--verification-csv`, `--residue-masses-file`, `--spec` / `--spec-file`, `--medium-output-dir` / `--high-output-dir`, `--lsh-assignments`, `--target-dir`.
