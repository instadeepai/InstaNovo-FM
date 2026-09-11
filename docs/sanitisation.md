# Sanitisation of ported code

The model, dataset pipeline and downstream code in this repository are ports of an internal
repository that runs on a private cluster. That repository's configs, manifests and run scripts
carry real infrastructure values, because they have to work. This document records what is
removed on the way here, so a reader can see what was changed rather than having to trust that
something was.

## What the gate rejects

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) fails the build if any of these appears
anywhere in the tree:

```
ait-pdfs   dtu-denovo-s-   mass-spec-foundation-model-
FOUNDATION_MODEL_AWS   id-platformhub   iam.gserviceaccount
```

The list is deliberately specific rather than a general "looks like a secret" heuristic: it names
the identifiers that actually exist upstream, so it cannot pass by accident and cannot be
weakened without the diff showing it. `tests/test_search_data.py` enforces the same rule on the
committed spreadsheets, which the scanner cannot read.

## Placeholders

| placeholder | replaces |
|---|---|
| `<inputs-bucket>`, `<outputs-bucket>`, `<fm-bucket>`, `<instanovo-bucket>` | the four cluster object-store buckets |
| `<data-root>` | the cluster mount points the pipeline reads from |
| `<run-id>` | experiment UUIDs in `s3://…/output/<uuid>/` paths |
| `<mlflow-tracking-uri>` | the internal MLflow server |
| `<gcp-project>`, `<service-account>` | the GCP project, its service account, and a key-file id |
| `<your-bucket>` | a bucket in an example that has no single right value |

Four buckets get four placeholders rather than one, so a config or run script still says *which*
store it means while naming none of them.

Two things are handled by rewriting rather than substitution:

- **MLflow** defaults to off, with an empty tracking URI, rather than pointing at the internal
  server. Standard `MLFLOW_*` variable *names* are documented here and are not secrets.
- **`data/search_data.xlsx`**'s `file path` column held UNC paths under an internal file server.
  Each is reduced to `<parent>/<filename>` — which is all `_extract_lookup_key` consumes, so
  behaviour is unchanged — and every parent is a public repository accession.

## What is not ported

Some things are dropped rather than scrubbed, because a placeholder would leave a file that looks
usable and is not:

- The internal `CLAUDE.md` files.
- `instanovo`'s own inference configs for models unrelated to this one.
- Cluster job manifests, and the scheduler invocations that reference them. Where a doc needs to
  describe a run, it describes the recipe.
- **`proteomics-mcp`.** The internal branch carries a copy of this package in-tree, and the
  retrieval and rescue tasks use it for their spectral-evidence metrics — the observed-versus-
  theoretical blocks that check whether a transferred peptide explains the query spectrum. It is
  left out here because it is unpublished work by its author rather than a released dependency: it
  is not on PyPI and its repository is not public. Its licence is permissive, so it can be vendored
  once that changes.

  Everything that does not need it still runs. `spectrum_metrics/` is self-contained numpy, and
  `mcp_scoring.py` guards the import behind `MCP_AVAILABLE`, so the retrieval, rescue and
  observed-versus-observed paths are unaffected. What cannot be reproduced here, until the package
  is published, is the evidence-metric blocks alone.

## Reviewing a port

The sanitisation is kept separable from the port so the two can be read independently. A porting
branch in the internal repository holds the same contents as its source with only the
substitutions above applied, so

```
git diff <source-branch>..<source-branch>-sanitised
```

is the sanitisation on its own, and the pull request here is then the port on its own. For the
downstream de novo work that pair is `259-baselines` and `259-baselines-sanitised`.
