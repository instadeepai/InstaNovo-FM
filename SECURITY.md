# Security policy

## Reporting a vulnerability

Report vulnerabilities in InstaNovo-FM's own code through GitHub's private
[security advisory](https://github.com/instadeepai/InstaNovo-FM/security/advisories/new)
form, not a public issue.

For vulnerabilities in a dependency, report them upstream. This document records
which ones we already know about.

## Why dependency versions are pinned, and not updated

**This repository exists to reproduce the results in the InstaNovo-FM manuscript.**
Its dependency versions are therefore pinned to the ones the published results were
produced with, taken from the development lockfile — not to the latest available, and
not to the latest patched.

That is a deliberate trade-off. Bumping a dependency to clear an advisory would mean
the code no longer reproduces the numbers in the paper, which is the one thing this
repository is for. A reader who cannot reproduce a figure cannot check the science.

The consequence is that a vulnerability scanner run against this project **will
report findings, and they are real**. They are enumerated below with an assessment of
whether they are reachable here, and the same assessment is published in
machine-readable form as [`vex.openvex.json`](vex.openvex.json) so scanners can
consume it directly.

**If you are deploying this code rather than reproducing the paper with it, do not
use these pins.** Take the newest versions your own testing supports.

Separately, note that installing this project fetches proprietary NVIDIA binaries;
that is a licensing matter rather than a security one, and it is covered in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Known advisories in pinned dependencies

Audited against [OSV](https://osv.dev/) on 2026-09-01, on the default install
(no optional extras). Reproduce with the snippet at the end of this file.

### `torch==2.8.0` — the version the model was trained with

| advisory | severity | fixed in |
|---|---|---|
| GHSA-vgrw-7cvw-pwgx — `unpack_sequence` memory corruption | MODERATE | 2.9.1 |
| PYSEC-2025-203 — DoS in `torch.linalg.lu` | — | 2.9.0 |
| PYSEC-2025-204 — unexpected behaviour in `torch.rot90` | — | 2.9.0 |
| PYSEC-2025-206 — integer overflow in `torch.nan_to_num` | — | 2.9.0 |
| PYSEC-2026-2286 | — | 2.10.0 |
| GHSA-qfhq-4f3w-5fph — `torch.lstm_cell` memory corruption | LOW | 2.10.0 |
| GHSA-rrmf-rvhw-rf47 — `torch.jit.script` memory corruption | LOW | 2.13.0 |
| PYSEC-2026-139 | — | not fixed |

**Assessment: not reachable as this code is used.** Every one of these requires
attacker-controlled input to a specific tensor operation. This project feeds torch
parquet files that it produced itself from PRIDE submissions, and exposes no network
service, no untrusted upload path and no user-supplied model code. None of the named
functions (`linalg.lu`, `rot90`, `nan_to_num`, `lstm_cell`, `jit.script`,
`unpack_sequence`) is called by this codebase.

**Why not simply move to 2.9.1 or later.** torch 2.8.0 is the version the published
model was trained and evaluated with, confirmed from the development lockfile. It is
also within the range `instanovo==1.2.2` declares support for (`>2.5,<2.9` on Linux
and Windows, `<2.8` on macOS); 2.9.x is outside it. Note that an earlier state of
this repository resolved torch 2.13.0, which is unaffected by all but one of the
above — that resolution was accidental, since instanovo's torch pins sit behind
extras this project does not select, so nothing enforced the ceiling.

### `pyarrow==22.0.0` — the version the corpus was built with

| advisory | severity | fixed in |
|---|---|---|
| GHSA-rgxp-2hwp-jwgg / PYSEC-2026-113 — use-after-free reading an IPC file with pre-buffering | HIGH | 23.0.1 |

**Assessment: reachable in principle, and worth understanding.** This is the one
advisory that touches a code path this project genuinely uses:
`scripts/preprocessing/convert_ipc_to_parquet.py` reads Arrow IPC files. The
mitigating facts are that the IPC files are produced by this same pipeline from
PRIDE-hosted data, are not accepted from third parties, and are read in a batch
conversion step rather than by any service.

**If you run this pipeline over IPC files you did not produce yourself, upgrade
pyarrow to 23.0.1 or later first.** That will not change the conversion output.

### `mlflow==3.15.2`

| advisory | severity | fixed in |
|---|---|---|
| GHSA-h7x2-h6g9-p789 — SSRF via unvalidated `api_base` in the AI Gateway | HIGH | not fixed |

**Assessment: component not used.** The advisory is in MLflow's AI Gateway, which
this project never starts. MLflow is present only as an experiment-tracking sink and
is **disabled by default** in the shipped configs. It also transitively caps
`cryptography` below its fix (see below).

### `cryptography==49.0.0`

| advisory | severity | fixed in |
|---|---|---|
| GHSA-g6cj-pr64-35w5 / PYSEC-2026-3552 — Bleichenbacher oracle in PKCS#7 `EnvelopedData` decryption | HIGH | 50.0.0 |

**Assessment: not reachable, and capped upstream.** Nothing here performs PKCS#7
decryption; cryptography arrives transitively via mlflow, which pins
`cryptography<50` and so blocks the fix regardless.

### `msgpack==1.1.2`

| advisory | severity | fixed in |
|---|---|---|
| GHSA-6v7p-g79w-8964 / PYSEC-2026-3625 — out-of-bounds read on `Unpacker` reuse after a caught error | HIGH | 1.2.1 |

**Assessment: not reachable, and pinned upstream.** No code here uses msgpack; it is
pinned exactly (`==1.1.2`) by `signalrcore`, itself a transitive mlflow dependency.

### `datasets==4.0.0`

| advisory | severity | fixed in |
|---|---|---|
| PYSEC-2026-3716 — path traversal | — | 5.0.1 |

**Assessment: not reachable as used.** `datasets` is used to read local parquet the
pipeline produced. The traversal requires a maliciously crafted dataset archive, and
nothing here loads third-party dataset archives.

### `pytest==8.3.3` — development only

| advisory | severity | fixed in |
|---|---|---|
| GHSA-6w46-j5rx-g56g / PYSEC-2026-1845 — vulnerable `tmpdir` handling | MODERATE | 9.0.3 |

**Assessment: development dependency, not shipped.** It is in the `dev` group and is
absent from any runtime install. The advisory concerns predictable temporary
directories on multi-user machines.

## What is deliberately *not* pinned

`cuml-cu12` (RAPIDS) is **not** a locked dependency. Locking it caps
`pyarrow<19.0.0a0` through `cudf-cu12`, and because `uv.lock` is a universal
resolution that cap applies to every install — it silently pulled `pyarrow` back from
22.0.0 to 18.1.0 and added a hard `distributed==2024.11.2` pin carrying its own
advisory. Development kept RAPIDS out of the lockfile for the same reason. See
[`docs/gpu-probes.md`](docs/gpu-probes.md).

## Reproducing this audit

```bash
uv sync --group dev
uv run python - <<'PY'
import importlib.metadata as md, json, urllib.request
pkgs = sorted({(d.metadata["Name"], d.version) for d in md.distributions()
               if d.metadata.get("Name") and d.version})
q = [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in pkgs]
for i in range(0, len(q), 200):
    chunk = q[i:i + 200]
    req = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": chunk}).encode(),
        headers={"Content-Type": "application/json"},
    )
    for query, result in zip(chunk, json.load(urllib.request.urlopen(req)).get("results", [])):
        for v in result.get("vulns") or []:
            print(query["package"]["name"], query["version"], v["id"])
PY
```

`pip-audit` also works, but note that it re-resolves the requirement set and will
fail on GPU-only packages that live on NVIDIA's index rather than PyPI.
