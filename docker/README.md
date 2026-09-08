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

## Licences

Neither baseline is vendored here. Each image builds its dependency from
upstream at a pinned version, so what follows is what those upstreams are
licensed under, not a licence this repository grants. The adapters under
`src/instanovo_fm/eval/` are ours and are Apache-2.0 like the rest of the
repository.

| what the image pulls in | version pinned | licence |
|---|---|---|
| [Casanovo](https://github.com/Noble-Lab/casanovo) | `casanovo==5.1.2` from PyPI | Apache-2.0 |
| Casanovo checkpoint `casanovo_v5_0_0.ckpt` | the `v5.0.0` release asset | see below |
| [MassNet-DDA](https://github.com/guomics-lab/MassNet-DDA) (XuanjiNovo) | commit `84105ec` | Apache-2.0, © 2024 PHOENIX center |
| [`ctcdecode`](https://github.com/parlance/ctcdecode) | the copy in that MassNet-DDA commit | MIT |
| [`imputer-pytorch`](https://github.com/rosinality/imputer-pytorch) | the copy in that MassNet-DDA commit | MIT |
| [CuPy](https://github.com/cupy/cupy) | `cupy-cuda12x==13.6.0` | MIT |

GitHub reports MassNet-DDA's licence as "Other" because its `LICENSE` is the
short-form Apache notice rather than the full text. It is Apache-2.0: read the
file at the pinned commit if you need to confirm that for yourself.

Two things the table cannot tell you.

**The model weights are not the code.** Both images fetch a trained checkpoint,
and neither upstream states terms for its weights separately from its source. The
Casanovo checkpoint comes from a GitHub release asset; XuanjiNovo's is fetched
separately at run time rather than baked in. If you intend to redistribute either
checkpoint, or anything derived from one, that is worth asking the authors about
rather than inferring from the repository licence.

**`Dockerfile.xuanjinovo` builds on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`,**
which carries CUDA and cuDNN under NVIDIA's own terms, not an open-source licence.
The same applies to a default install of this project, and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) sets out what that means —
it matters most if you push either image to a registry, because then you are
redistributing NVIDIA's binaries under NVIDIA's terms.

Every third-party baseline import in `src/instanovo_fm/eval/` is inside a
function, never at module scope. That is what lets the adapters sit on `main`:
importing `instanovo_fm.eval` does not require Casanovo or the MassNet stack.
There is a test for it, so it stays true.
