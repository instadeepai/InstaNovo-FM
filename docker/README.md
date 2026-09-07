# Baseline environments

The external baselines cannot share InstaNovo-FM's environment, so each gets an
image. The *adapter* code lives on `main` under `src/instanovo_fm/eval/` and is
linted and type-checked with everything else; only the incompatible third-party
dependencies are confined here.

| image | why it is separate |
|---|---|
| `Dockerfile.casanovo` | Casanovo pins `numpy<2.0`; instanovo requires `numpy>=2.0.2`. Irreconcilable, so Casanovo gets its own interpreter. |
| `Dockerfile.xuanjinovo` | Needs CUDA 12.1 + torch 2.1, `ctcdecode`, `imputer-pytorch` and `cupy`, and fetches the upstream model from `guomics-lab/MassNet-DDA` at a pinned commit. None of it is vendored here. |

Both write embeddings in the HDF5 layout `instanovo_fm.eval.embedding_io` reads,
so the linear-probe, retrieval and UMAP tasks run on identical downstream code
regardless of which encoder produced the embeddings.

Every third-party baseline import in `src/instanovo_fm/eval/` is inside a
function, never at module scope. That is what lets the adapters sit on `main`:
importing `instanovo_fm.eval` does not require Casanovo or the MassNet stack.
There is a test for it, so it stays true.
