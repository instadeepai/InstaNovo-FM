# Third-party notices

The source code in this repository is licensed under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). **That licence does not extend to the third-party binary
packages that installing this project fetches**, some of which are proprietary.

This file records the ones whose terms are more restrictive than Apache-2.0, so
that anyone redistributing a built environment or container image knows what they
are redistributing.

## NVIDIA CUDA runtime and kernel libraries — proprietary

> [!IMPORTANT]
> These are installed by a plain `uv sync`, **not** by an optional extra. The chain
> is `instanovo` → `accelerate>=1.6.0` → `torch` → these wheels, which PyTorch
> declares for `platform_system == "Linux"` unconditionally. So a default Linux
> install of this project pulls closed-source NVIDIA binaries, even though nothing
> here asks for a GPU and neither `torch` nor any `nvidia-*` package is declared
> directly in `pyproject.toml`.

Twelve of the fourteen `nvidia-*` wheels installed by a default Linux sync declare
an NVIDIA proprietary licence in their wheel metadata: `nvidia-cudnn-cu12` as
`License-Expression: LicenseRef-NVIDIA-Proprietary`, the other eleven as
`License: NVIDIA Proprietary Software`. Each ships NVIDIA's terms as `License.txt`
inside its `.dist-info` directory — read that file for the governing terms, and note
that NVIDIA distributes cuDNN under a separate cuDNN Software License Agreement.

| package | provides | licence |
|---|---|---|
| `nvidia-cublas-cu12` | cuBLAS dense linear algebra — backs most matmuls, including attention | proprietary |
| `nvidia-cudnn-cu12` | cuDNN deep-learning primitives — backs the fused attention kernels | proprietary |
| `nvidia-cusparselt-cu12` | cuSPARSELt structured sparse matmul | proprietary |
| `nvidia-cusparse-cu12` | cuSPARSE sparse linear algebra | proprietary |
| `nvidia-cusolver-cu12` | cuSOLVER dense and sparse solvers | proprietary |
| `nvidia-cufft-cu12` | cuFFT fast Fourier transforms | proprietary |
| `nvidia-curand-cu12` | cuRAND random number generation | proprietary |
| `nvidia-cufile-cu12` | GPUDirect Storage I/O | proprietary |
| `nvidia-cuda-runtime-cu12` | CUDA runtime | proprietary |
| `nvidia-cuda-nvrtc-cu12` | NVRTC runtime compiler | proprietary |
| `nvidia-cuda-cupti-cu12` | CUPTI profiling interface | proprietary |
| `nvidia-nvjitlink-cu12` | JIT link-time optimisation | proprietary |
| `nvidia-nccl-cu12` | NCCL collective communication (multi-GPU training) | BSD-3-Clause |
| `nvidia-nvtx-cu12` | NVTX profiling annotations | Apache 2.0 |

The last two are **not** proprietary, and that distinction is version-dependent
rather than stable: under torch 2.13, which resolved before torch was pinned to
match instanovo, the equivalent `nvidia-nccl-cu13` declared
`LicenseRef-NVIDIA-Proprietary`, and an `nvidia-nvshmem-cu13` package appeared that
has no cu12 counterpart. Re-check after any dependency bump rather than trusting
this table.

## Attention kernel backends

`src/instanovo_fm/model/attention/` calls
`torch.nn.functional.scaled_dot_product_attention` and lets PyTorch choose a
backend at run time. On NVIDIA hardware the kernel that actually executes is
closed source — cuDNN or cuBLAS from the list above.

To be clear about what this repository does and does not contain: it has **no
dependency on the `flash-attn` package** and vendors **no CUDA kernel source** of
its own. The attention modules are plain PyTorch, and `flash.py` is named for the
backend PyTorch may select, not for a bundled implementation. Selecting the
backend is PyTorch's decision, and it is the only place proprietary kernels enter
the model's forward pass.

## Optional GPU extras

The `gpu` extra adds `cuml-cu12` (RAPIDS cuML) for GPU linear probes, which also
pulls `cupy-cuda12x` and `nvidia-nvcomp-cu12`, and requires NVIDIA's own package
index.

`cuml-cu12` declares `Apache-2.0` and `cupy-cuda12x` declares `MIT`, so those two
are permissively licensed **as source** — but both are binary builds linked
against the proprietary CUDA libraries above. Every evaluation task that uses cuML
skips cleanly when it is absent, so this extra is never required.

`triton`, installed by PyTorch, is MIT-licensed. It compiles to NVIDIA PTX but is
not itself proprietary.

## Avoiding the proprietary packages entirely

A CPU-only environment needs none of them. Note that `instanovo`'s own `cpu`
extra does **not** achieve this — it only constrains the torch *version* range,
not which wheel variant is fetched. Avoiding the NVIDIA wheels means taking torch
from PyTorch's CPU index:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

The dataset pipeline, the figure notebooks and the release scripts do not need a
GPU at all. Training and the embedding-evaluation harness are the parts that do.

## How to regenerate this list

```bash
uv run python - <<'PY'
import pathlib, re
for d in sorted(pathlib.Path(".venv/lib/python3.11/site-packages").glob("*.dist-info")):
    m = (d / "METADATA")
    if not m.exists():
        continue
    t = m.read_text(errors="ignore")
    name = re.search(r"^Name: (.+)$", t, re.M)
    lic = re.search(r"^License-Expression: (.+)$", t, re.M) or re.search(r"^License: (.+)$", t, re.M)
    if name and re.match(r"nvidia-|cuml|cupy|triton", name.group(1)):
        print(name.group(1), "|", lic.group(1) if lic else "none declared")
PY
```

Re-run it after any dependency bump. PyTorch changes both its bundled CUDA lineage
and those wheels' licence metadata between releases: pinning torch to instanovo's
tested range moved this project from the `cu13` lineage to `cu12`, dropped
`nvidia-nvshmem` entirely, and turned NCCL from proprietary into BSD-3-Clause.
