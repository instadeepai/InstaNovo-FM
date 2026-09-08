# Upstream XuanjiNovo de novo benchmark

Runs the **upstream** XuanjiNovo model — published in the
[MassNet-DDA](https://github.com/guomics-lab/MassNet-DDA) repo, CTC beam search +
Precise-Mass-Control — on our validation parquets and scores it with the *same* scorer as
InstaNovo / Casanovo, so the numbers are directly comparable.

The upstream model needs its own Docker image (torch 2.1 / cu12.1, `ctcdecode`, `imputer-pytorch`),
which cannot be merged into `docker/Dockerfile.xuanjinovo`. So the benchmark is **three steps across two
images**, not a single job:

| # | Step | Image | Where |
|---|------|-------|-------|
| 1 | parquet → MGF + targets sidecar | the InstaNovo-FM environment | any host with the InstaNovo env + access to the validation parquets |
| 2 | XuanjiNovo inference → `denovo.tsv` | `Dockerfile.xuanjinovo` | a single CUDA GPU (we used one 80GB H100) |
| 3 | join + score → prediction CSV, `summary.json`, results row | the InstaNovo-FM environment | anywhere with the InstaNovo env |

**Paths.** Examples below use local paths for readability. Every path argument also accepts an
`s3://` URI, with two exceptions noted in [Gotchas](#gotchas). S3 access goes through
`instanovo.utils.s3.S3FileHandler`, which needs `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`.

---

## 1. parquet → MGF

```bash
uv run python -m instanovo_fm.eval.convert_parquet_to_mgf \
  --datasets gluc=$VALIDATION_ROOT/biological/annotated/dataset-gluc-*.parquet \
             immuno=$VALIDATION_ROOT/biological/annotated/dataset-immuno-*.parquet \
  --mgf_path /data/biological.mgf \
  --sidecar_path /data/biological_targets.parquet
```

`$VALIDATION_ROOT` is wherever the annotated validation parquets live.

- Use `--parquet_path <glob>` for a single dataset instead of `--datasets name=path …`.
- `--datasets` gives composite `<dataset>:<index>` ids and a per-dataset `group`, which is what makes
  the per-group metric breakdown in step 3 work. Prefer it — one job for all species.
- **Keep the sidecar.** It holds the targets and is the only way to join predictions back in step 3.
- Rows the model structurally cannot handle (charge outside `[1, 10]`, targets using residues it
  cannot emit) are dropped here by the shared `filter_unpredictable_rows`, identically to the other
  baselines. `--max_samples N` caps spectra per dataset for smoke tests.

## 2. Inference

```bash
docker build -f Dockerfile.xuanjinovo -t xuanjinovo .

docker run --gpus all -v /data:/data xuanjinovo \
  python -m instanovo_fm.eval.run_xuanjinovo /data/biological.mgf /data/xuanjinovo_out
```

Both args are positional: `<input_mgf> <output_prefix>`. The runner writes `denovo.tsv` plus run logs
to the output prefix; either arg may be a local path or an `s3://` URI.

Three directories are used, and only one is ever deleted:

| Directory | Default | Lifetime |
|-----------|---------|----------|
| input MGF downloads | `data/` | written into, never deleted |
| checkpoint downloads | `checkpoints/` | written into, never deleted |
| upstream's output | `/tmp/xuanjinovo` | **wiped at the start of every run** |

The two download directories are relative, so they resolve under the image `WORKDIR`
(`/app`) in a container and under the invocation directory locally — both are already in
`.gitignore`. A container's filesystem does not persist between runs, so downloads simply
repeat there; locally they persist and the checkpoint is reused. The input MGF is always re-downloaded even when present, because step 1
regenerates it under the same filename whenever the dataset changes — reusing it by name is how you
silently benchmark stale data. A downloaded checkpoint is named after its source URL, so switching
`XUANJINOVO_CKPT_URL` cannot reuse the previous one.

Sources that are already local paths are read where they are. The write overwrites whatever is at
the output prefix.

S3 paths need `AWS_ENDPOINT_URL` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`; an `s3://` argument
without them fails immediately rather than silently reading or writing nothing. Because the container
is ephemeral, a `denovo.tsv` that does not land at the destination fails the run — set
`XUANJINOVO_ALLOW_LOCAL_FALLBACK=true` to keep the local copy and warn instead.

Although the runner is an `instanovo` module, the InstaNovo package is **not** installed in this
image — upstream pins torch 2.1 while InstaNovo requires >2.5. `Dockerfile.xuanjinovo` copies in only
the runner and its import closure (`instanovo.utils.{s3,checkpoints,colorlogging}`, `instanovo.constants`)
and sets `PYTHONPATH=/xuanjinovo`. `TestXuanjiNovoImageClosure` fails the build if that set drifts
from what the runner imports, or if any copied module grows a torch/pandas/numpy import.

The image sets `ENTRYPOINT []` to reset upstream's `python -m XuanjiNovo.XuanjiNovo`, so the command
above runs verbatim rather than being appended to upstream's entrypoint.

Env-only knobs, all optional (each also has a matching `--flag`):

| Var | Default | Note |
|-----|---------|------|
| `XUANJINOVO_CKPT_S3` | — | checkpoint URI in object storage; **prefer this** over re-pulling from HF each run |
| `XUANJINOVO_CKPT_URL` | HF `XuanjiNovo_100M_massnet.ckpt` | download-URL fallback |
| `XUANJINOVO_N_BEAMS` | `40` | upstream recommendation |
| `XUANJINOVO_MASS_CONTROL_TOL` | `0.1` | PMC tolerance, Da |
| `XUANJINOVO_BATCH_SIZE` | `64` | |
| `XUANJINOVO_GPU` | `0` | comma-separated GPU ids |
| `XUANJINOVO_ALLOW_LOCAL_FALLBACK` | `false` | keep the local copy instead of failing when the write does not land |
| `XUANJINOVO_DATA_DIR` | `data` | where downloaded input MGFs are written |
| `XUANJINOVO_CKPT_DIR` | `checkpoints` | where downloaded checkpoints are written |
| `XUANJINOVO_WORK_DIR` | `/tmp/xuanjinovo` | upstream's output; wiped every run |

## 3. Score

```bash
uv run python -m instanovo_fm.eval.score_xuanjinovo \
  --tsv_path /data/xuanjinovo_out/denovo.tsv \
  --sidecar_path /data/biological_targets.parquet \
  --output_csv /data/results/xuanjinovo_biological.csv \
  --output_dir /data/results/xuanjinovo_summary \
  --results_csv /data/results/denovo_benchmark_biological.csv \
  --run_name xuanjinovo_biological --num_beams 40
```

Emits the canonical prediction CSV, a `summary.json` with overall **and** per-group blocks, and
appends a wide results row in `predictor.py` format so it drops straight into the benchmark table.

Metrics come in two denominators: unsuffixed = **true** denominator (spectra filtered in step 1 and
spectra the model returned nothing for count as misses — the honest number for cross-model
comparison); `*_predictable` = filtered denominator.

---

## Reproducibility

`Dockerfile.xuanjinovo` pins the MassNet-DDA repo via `ARG MASSNET_COMMIT`, currently
`84105ec5d871efd4a8aa98e79b859ffd927d78cc` (`main` HEAD on 2026-07-21).

## Gotchas

- **Notation.** Upstream's `denovo.tsv` writes *bracketed* offsets (`C[+57.021]`) → `XUANJINOVO_BRACKETED_TO_UNIMOD`.
- **Join key.** The converter encodes it as `TITLE=pid=<id>`, which upstream passes through verbatim.
  If you regenerate the MGF you must regenerate the sidecar with it — ids are positional into the
  *filtered* frame.
- **Reused output directories.** XuanjiNovo appends a fresh `%Y%m%d%H%M%S` subdirectory to `--output`
  on every invocation and never clears the parent, so re-running in one container would accumulate
  one subdirectory per run. The runner wipes its output directory up front, so this run's is the only
  one present; finding more than one afterwards is a hard error rather than a silent choice between a
  fresh and a stale result.
- The converter's `--mgf_path` and `--sidecar_path` are **local-only** (its `--parquet_path` /
  `--datasets` inputs do accept `s3://`). If the inference host reads from object storage, upload the
  MGF yourself after step 1, and keep the sidecar somewhere step 3 can reach.

## Tests

```bash
uv run pytest tests/unit_test/foundational/test_xuanjinovo.py
```

---

## Running it

The three steps run in different environments; see `docker/README.md`. Steps 1
and 3 need only the InstaNovo-FM environment. Step 2 needs
`docker/Dockerfile.xuanjinovo`, which builds the upstream MassNet-DDA source at a
pinned commit.

Object-storage arguments accept `s3://` URIs and read their endpoint and
credentials from the standard `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` variables, so any S3-compatible store works. Local paths
work throughout except where noted in [Gotchas](#gotchas).
