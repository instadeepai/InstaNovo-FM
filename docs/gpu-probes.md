# GPU-accelerated linear probes (cuML)

The embedding-evaluation harness can run its linear probes on GPU through RAPIDS
cuML. **cuML is deliberately not a locked dependency**, and the probes fall back to
scikit-learn when it is absent, so nothing here is required to reproduce the
manuscript's results.

## Why it is not in `pyproject.toml`

RAPIDS wheels carry constraints that are incompatible with a single resolved
environment. Concretely, `cuml-cu12` pulls `cudf-cu12`, which caps
`pyarrow<19.0.0a0`. Because `uv.lock` is a *universal* resolution across every
extra, that cap applies to **every** install of this project, not only the ones
that select the extra — so locking cuML silently pinned `pyarrow` to 18.1.0 where
the rest of the project resolves 22–25. RAPIDS also pins `distributed` exactly and
caps `pytest` at 7.x.

Development handled this the same way, keeping RAPIDS out of the lockfile and
installing it as a separate step. This repository follows that, for the same reason
`casanovo` is installed separately (see the note beside the dependency list).

## Installing it

Mirrors the development environment, including the version used there:

```bash
uv pip install cuml-cu12==26.4.0 --extra-index-url https://pypi.nvidia.com
```

cuML's installation upgrades some NVIDIA runtime libraries in place, which can
break the ones PyTorch expects. Development force-reinstalled PyTorch's CUDA
runtime afterwards to repair that, so do the same if imports start failing:

```bash
uv pip install --reinstall torch
```

## Confirming the fallback

Without cuML, the probe tasks should run on scikit-learn rather than fail:

```bash
uv run pytest tests/unit_test/foundational/test_linear_probe.py -q
```

Note that installing the optional extras exposes test failures in these paths
(`evoc`, `glass-box-umap`, `cuml`) that do not occur without them; they are not
exercised by CI, which installs no extras.
