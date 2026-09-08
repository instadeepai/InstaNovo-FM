# Data audits

Everything needed to reproduce the manuscript's three **data-audit** supplementary
tables — the ones describing the released corpus itself, labelled
`stab:filter_attrition`, `stab:charge_acquisition` and `stab:modifications`. They are
**generated**, not transcribed: `make_data_audit_tables.py` reads the audit JSON and
emits the LaTeX, so a number in the manuscript traces back to the scan that produced
it. Referenced by label rather than by number, since numbering shifts as other
supplementary tables change.

`--mount` is required and has no default on every script here. The path is
deployment-specific, and a hard-coded one would go stale.

## Running

```bash
ROOT=/path/to/tier/directories        # holds {hcfm,mcfm,lcfm}_splits and {hcfm,mcfm,lcfm}

python scripts/release/audit/verify_filter_conformance.py   --mount "$ROOT" --out-dir audit_out
python scripts/release/audit/charge_acquisition_crosstab.py --mount "$ROOT" --out-dir audit_out
python scripts/verification/inventory_modifications.py      --mount "$ROOT" \
    --output-csv audit_out/modifications.csv

# needs no mount: checks the recorded code masses against UNIMOD
python scripts/release/audit/check_unimod_coverage.py \
    --codes scripts/release/audit/in_code_masses.csv \
    --out audit_out/unimod_coverage.json

python scripts/release/audit/make_data_audit_tables.py \
    --audit-dir audit_out --unimod-coverage audit_out/unimod_coverage.json \
    --out data_audit_tables.tex
```

The generated LaTeX needs `booktabs`, `longtable` and a `\mz{}` macro, all already in
the manuscript preamble.

## What each one answers

| script | question | table |
|---|---|---|
| `verify_filter_conformance.py` | Do the published splits satisfy the five retention criteria, and how many rows does each criterion remove? | `stab:filter_attrition` |
| `charge_acquisition_crosstab.py` | Why do spectra with precursor charge 0 exist? | `stab:charge_acquisition` |
| `inventory_modifications.py` | What modifications does the corpus contain, and which are excluded? | `stab:modifications` |
| `check_unimod_coverage.py` | Does any `[IN:n]` code in fact have a UNIMOD identifier? | `stab:in_codes` |
| `census_release.py` | File counts, sizes, row counts, shard totals and per-config schemas. Not a table; it is what the release manifests are built from. | — |

`census_release.py` is the slowest (two passes, ~15 min over 46k files); the other
scans read only the few columns they need, and `check_unimod_coverage.py` touches no
data at all -- it reads `in_code_masses.csv` and queries pyOpenMS, so it runs anywhere in
a second.

## `in_code_masses.csv`

One row per `[IN:n]` code: the code, the residue it is observed on, its monoisotopic
mass delta in Da, and the number of spectra carrying it.

The masses are the curated values that the preprocessing pipeline uses to compute
fragment masses for these modifications; they were derived from annotated glycan
compositions. The compositions themselves are not part of this repository, which is why
`check_unimod_coverage.py` can compare masses but cannot confirm composition identity.
The spectra counts are measured by `inventory_modifications.py`.

Two things in the file are worth knowing. `IN:3174` sits on lysine and is a
ubiquitination remnant rather than a glycan, so it is the one code whose UNIMOD search
runs against a different residue. And the codes are a closed vocabulary of 175: any code
outside it appearing in future data means the annotation pipeline has grown a case this
table does not cover, which `inventory_modifications.py` would report as an unrecognised
token.

## Notes that cost time to establish

**A skipped file must never produce a passing verdict.** `verify_filter_conformance.py`
reports `INCONCLUSIVE`, not `CONFORMS`, if any file was unreadable or no rows were read.
An early version hit a column-selection bug, read zero rows from every file, and
reported conformance — zero rows trivially satisfy every criterion.

**Mass tables mix two conventions.** The recorded masses for these codes are
modification *deltas*, but tables of this kind often store a residue's *total* mass
(residue plus modification) for ordinary modifications while storing bare deltas for
glycans. Reading one as the other shifts every lookup by a residue mass -- about
114~Da for Asn -- and turns real UNIMOD matches into apparent absences.
`check_unimod_coverage.py` therefore runs a control over modifications whose identifier
is known, and refuses to report if any is not recovered. The control has already earned
its keep: `getUniModRecordId()` returns `-1` for entries that are not UNIMOD records,
and because `-1` is truthy, a naive presence test silently indexes every modification
pyOpenMS knows rather than only the UNIMOD ones.

**The audits exit non-zero when they *find* something.** Chain them with `;` rather than
`&&`, or a finding in one will stop the next from running.

**Compare Arrow logical schemas, not physical ones.** `census_release.py` reports the
physical parquet schema, where a list column such as `mz_array` surfaces as
`element:FLOAT`. Differences in that field are usually list-encoding artefacts rather
than real divergence.

**The five criteria are very unevenly load-bearing.** Retention time and the
unresolved-modification criterion do essentially all the work; the charge and precursor
*m/z* bounds match no row in the corpus at all. Worth knowing before treating a
reimplementation that omits one as equivalent.
