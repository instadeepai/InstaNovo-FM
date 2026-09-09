# Dataset release scripts

Two artefacts go to two HuggingFace repos, and the scripts are deliberately separate:

| artefact | repo | scripts |
|---|---|---|
| the labelled spectra corpus, in tiers and flavours | [`InstaDeepAI/InstaNovo`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo) | `hf_probe.py`, `hf_upload.py`, `dataset_card.md` |
| the frozen spectrum embeddings | [`InstaDeepAI/InstaNovo-FM-embeddings`](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-FM-embeddings) | `prepare_embeddings.py`, `hf_upload_embeddings.py`, `embeddings_card.md` |

They share no tier logic and carry different licences, so folding the embeddings into
`hf_upload.py` would have meant a flag that changes what every other flag means. The
embeddings runbook is in the header of `hf_upload_embeddings.py`; the rest of this file
is about the corpus.

Upload the labelled confidence tiers to the HuggingFace dataset repo.

## Order of operations

```bash
export INSTANOVO_FM_HF_TOKEN=...     # token with write access to the target repo
                                     # (INSTANOVO_HF_TOKEN is still read as a fallback;
                                     #  the probe prints which name it used)

# 1. Check the environment before moving any data. Separates "no network" from
#    "bad token" from "no write permission", which a failed upload would conflate.
python scripts/release/hf_probe.py --source-root "$SOURCE_ROOT"

# 2. See what would be sent. Both flavours of the tier, no upload.
python scripts/release/hf_upload.py --source-root "$SOURCE_ROOT" --tier hcfm --dry-run

# 3. Upload smallest tier first, so a configuration problem surfaces cheaply.
#    Each command sends both flavours of that tier.
python scripts/release/hf_upload.py --source-root "$SOURCE_ROOT" --tier hcfm
python scripts/release/hf_upload.py --source-root "$SOURCE_ROOT" --tier mcfm
python scripts/release/hf_upload.py --source-root "$SOURCE_ROOT" --tier lcfm

# 4. Confirm, without sending anything.
python scripts/release/hf_upload.py --source-root "$SOURCE_ROOT" --tier lcfm --verify-only
```

`--source-root` is required and has no default. The path is deployment-specific,
and hard-coding one would bake a stale location into a public repository.

## Two flavours per tier, and why both are published

Each tier exists on disk in two forms, and both are uploaded, because the
manuscript relies on each for a different claim.

| on disk | in the repo | contents |
|---|---|---|
| `<tier>_splits` | `<tier>/splits/` | `train`/`test`/`valid` parquet, quality-filtered and shuffled |
| `by_project/<tier>` | `by_project/<tier>/` | the tier before filtering and splitting, one directory per accession |

**`splits/` is what the model consumed** — the "training and evaluation data" of
Data Availability. Five quality filters are applied: retention time <= 10800 s, lower
isolation offset <= 300 Da, precursor charge 0-7 inclusive (**0 is kept**), precursor
*m/z* <= 2000, and no modification annotation unresolvable to a UNIMOD identifier
(`[IN:<digits>]` — mostly N-glycans, but not exclusively). Nulls pass every numeric
condition. Use this to reproduce the paper.

**`by_project/` is the input those splits were derived from.** It is published
because the filtering is lossy: rows the quality gates drop are not recoverable
from the splits. Without it, the peptide registry cannot be used to re-derive the
partitions or to extend them to new data, which Data Availability says it can.
Use this to re-split, or to apply different quality criteria.

The resulting layout:

```
splits/lcfm/         by_project/lcfm/
splits/mcfm/         by_project/mcfm/
splits/hcfm/         by_project/hcfm/
peptide_registry.parquet
```

`--flavour` defaults to `both`. Pass `--flavour splits` or
`--flavour by_project` to send one form on its own, which is useful when
uploading them in separate sessions.

## ACFM is not uploaded

`--tier acfm` is rejected rather than silently attempted. As Data Availability
states, the unlabelled tier is approximately 5.7 TB and is not redistributed; it
comprises every MS/MS scan from the same raw files, so it is reconstructible from
the accessions with the conversion pipeline deposited at Figshare.

## The dataset card

`dataset_card.md` is the card for the HuggingFace repo. It documents the
`splits/` vs `by_project/` distinction for people who arrive at the data without
reading this repo, so it should go up with the first tier rather than after:

```python
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="scripts/release/dataset_card.md",
    path_in_repo="README.md", repo_id="InstaDeepAI/InstaNovo", repo_type="dataset",
)
```

Check for an existing `README.md` on the repo first and reconcile rather than
overwrite. Its `license:` field is a placeholder to confirm before the repo goes
public.

## Making it fast

The upload is roughly a terabyte, so throughput is worth setting up deliberately.
Four levers, in descending order of effect.

**1. Give the job CPUs.** `upload_large_folder` is already a parallel worker pool
(hashing, upload-mode queries, chunk uploads), but when `num_workers` is unset
huggingface_hub picks `max(cpu_count // 2, 1)`. A 2-CPU container therefore gets
**one worker** and the upload is effectively serial. This script defaults to
`max(cpu_count, 8)` instead and prints what it chose, so the number is visible
rather than inferred. Size the job accordingly — 16 CPUs, not 2.

**2. Keep Xet high-performance on.** The target repos are Xet-backed
(`xetEnabled: true`), so transfers are chunked and content-addressed. The script
sets `HF_XET_HIGH_PERFORMANCE=1` before importing `huggingface_hub`, which is when
that variable is read. Do **not** reach for `hf_transfer`: it is no longer used, and
huggingface_hub emits a `FutureWarning` pointing at this variable instead.

**3. Shard across jobs.** The six prefixes are disjoint, so
`--tier X --flavour Y` can run as six concurrent jobs. Each call serialises its own
commits internally, and separate processes committing to the same branch can
conflict — but a failed commit re-queues its files and is retried, so parallel jobs
cost occasional wasted commit attempts, not data. Start with three concurrent jobs
and watch the reports before going wider.

**4. Optionally repack the per-project tiers.** Half the bytes sit in 99% of
the files: `splits` is 509 files for 531 GB, `by_project` is 45,858 files for
488.6 GB — averaging 0.7 MB per file for HCFM and 3.3 MB for MCFM. Every one of
those needs a hash, an upload-mode query and a commit entry, so per-file overhead
dominates that half of the upload. Consolidating them into ~1 GB parquet files
would take the repo from ~46,000 files to ~1,000. This breaks no HuggingFace limit
either way (the recommendation is <100k files per repo and <10k entries per folder,
and we are inside both), so it is an optimisation, not a prerequisite -- and one that
costs a full 488 GB read-and-write pass on the mount before the upload even starts,
so it likely does not pay for itself on upload time alone. The better arguments for
it are about the published dataset: 0.7 MB parquet files stream poorly, and an
earlier corpus audit found near-empty prefixes in this store, so some of those
15,286 files per tier may be empty.

**Repack within each project accession directory, never across it.** The
per-project tiers are laid out by accession, so merging across projects would take
away per-project selection. Merging inside one keeps
`by_project/lcfm/PXD012345/` selectable while still collapsing ~186 files to a
handful.

### Naming, if you do repack

Split files use the Hub's sharding convention,
`{split}-{index:05d}-of-{total:05d}.parquet`, with the canonical split name
`validation` rather than the current `valid`.

Note this is **not** needed for split detection: `datasets` already recognises
`valid` as an alias (`SPLIT_KEYWORDS[validation] = ['validation', 'valid', 'dev',
'val']`) and its pattern `**/valid[-._ 0-9]*` matches today's `valid_0.parquet`,
since `_` is in the separator class. Three things do improve:

- `-of-{total}` makes an incomplete set self-evident.
- Zero-padding sorts correctly. Today's names do not: `train_10` sorts before
  `train_2`, which affects this uploader and any glob-based consumer.
- It matches what `push_to_hub` produces, so the dataset looks native.

It also settles an existing inconsistency: the registry writes the split as
`validation` while the files on disk say `valid`.

**Do not apply the convention to `by_project/`.** It has no train, validation or
test split, and any of those keywords in a filename would make the Hub advertise a
split that does not exist. Use `data-{index:05d}-of-{total:05d}.parquet` there.
Neither `data` nor `by_project` is a split keyword, so nothing is auto-detected
from them; `dataset_card.md` declares every tier and flavour explicitly in its
`configs:` block instead of leaving it to inference.

Xet's deduplication is not worth counting on here: the two flavours hold the same
rows but are independently compressed and shuffled, so their bytes differ
completely. Dedup helps on retries and re-uploads, not across flavours.

## Resuming

`hf_upload.py` uses `upload_large_folder`, which is resumable: re-running the same
command skips files already present at a matching size. It then verifies every
local file against the remote tree by path and size, and exits non-zero listing
what is missing, so an interrupted run is safe to repeat until it reports
complete.

## After the upload

Tag the release and record both the tag and the commit SHA — a tag is a movable
pointer, the SHA is not:

```python
from huggingface_hub import HfApi
HfApi().create_tag("InstaDeepAI/InstaNovo", tag="v1.0", repo_type="dataset",
                   tag_message="Manuscript release")
```

Consumers pin it with `revision="v1.0"`. The manuscript's Data Availability has a
placeholder for the tag and SHA that needs filling in once this is done.
