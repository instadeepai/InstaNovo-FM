# Sample data

Three small MGF files, committed so the software can be demonstrated without downloading a dataset
first. [`scripts/demo.py`](../scripts/demo.py) runs on all three; see
[Demo](../README.md#demo) for what to expect back.

| file | spectra | size | what it is |
| --- | --- | --- | --- |
| `fig4_TGVELGKPTHFTVNAK.mgf` | 1 | 3.2 kB | the spectrum the manuscript's Fig. 4 is drawn from, annotated `TGVELGKPTHFTVNAK` |
| `retrieval_library.mgf` | 270 | 923 kB | a reference library: 30 tryptic peptides with replicate observations |
| `ninespecies_queries.mgf` | 20 | 43 kB | query spectra of peptides the library also contains, sharing no spectrum with it |

Together they are enough to exercise embedding, *de novo* sequencing and database-free retrieval.
They are **not** enough to train a model — see [Getting started](../docs/getting_started.md) for
that path.

## Provenance

All three are subsets of public data, taken from the
[InstaNovo-FM Hugging Face Space](https://huggingface.co/spaces/InstaDeepAI/InstaNovo-FM) that
demonstrates the same model, and vendored here so the demo runs offline.

- **`retrieval_library.mgf`** and **`ninespecies_queries.mgf`** come from the
  [nine-species benchmark](https://huggingface.co/datasets/InstaDeepAI/ms_ninespecies_benchmark):
  tryptic DDA spectra. The two files share no spectrum, so retrieving one against the other is a
  real search rather than a file matched against itself.
- **`fig4_TGVELGKPTHFTVNAK.mgf`** is rebuilt from the hero record committed at
  [`data/hero_000_TGVELGKPTHFTVNAK_y5+.json`](../data/hero_000_TGVELGKPTHFTVNAK_y5+.json), which
  [`notebooks/figure_4.ipynb`](../notebooks/figure_4.ipynb) draws Fig. 4 from. The intensities
  stored there are already sqrt-scaled and L2-normalised, so they are squared and scaled back up
  in the MGF; the processor's own sqrt, L2 and minimum-intensity steps undo that exactly, and the
  model sees the same 140 peaks the figure was made from.

Two caveats if you use these files for anything beyond the demo. Each spectrum is trimmed to its
200 most intense peaks — lossless for this model, whose encoder keeps 200 peaks anyway, but not a
faithful copy of the deposited raw data. And the peptide labels are the output of our FragPipe
reprocessing, not the depositors' own assignments. Cite the original accessions rather than this
directory.
