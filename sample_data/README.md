# Sample data

`spectra.mgf` is a single real MS/MS spectrum, small enough to commit, so the software can be
demonstrated without downloading a dataset first. See
[Demo](../README.md#demo) for what to run on it and what to expect back.

| | |
|---|---|
| Spectra | 1 |
| Annotation | `SEQ=IAHYNKR`, precursor *m/z* 451.25348, charge 2+ |
| Peaks | 25 |
| Size | 1,044 bytes |

## Provenance

Taken verbatim from [`sample_data/spectra.mgf`](https://github.com/instadeepai/InstaNovo/blob/main/sample_data/spectra.mgf)
in the [InstaNovo](https://github.com/instadeepai/InstaNovo) repository, which is licensed under
[Apache-2.0](https://github.com/instadeepai/InstaNovo/blob/main/LICENSE) — the same licence as this
repository's code. It is vendored rather than fetched so that the demo runs offline.

One spectrum is enough to exercise loading, preprocessing, encoding and *de novo* sequencing. It is
**not** enough for the retrieval demo or for the small training run in
[Getting started](../docs/getting_started.md), both of which need many spectra; those use the
larger inputs named in their own instructions.
