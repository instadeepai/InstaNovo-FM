#!/usr/bin/env bash
# Create the pinned environment for the figure notebooks and register it as a
# Jupyter kernel. Run once, from the repository root:
#
#     ./setup_kernel.sh
#
# then open a notebook and select the "InstaNovo-FM (figures)" kernel.
#
# Dependencies live in pyproject.toml under the "figures" dependency group. The
# notebooks deliberately do not pip install anything themselves: figure_3's
# zoom-region search is sensitive to the numpy build, and installing from inside
# a running kernel cannot change an already-imported module in any case.
set -euo pipefail

PYTHON_VERSION=${PYTHON_VERSION:-3.11}
KERNEL_NAME=${KERNEL_NAME:-instanovo-fm-figures}
KERNEL_LABEL=${KERNEL_LABEL:-"InstaNovo-FM (figures)"}

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
error: uv is required.

  curl -LsSf https://astral.sh/uv/install.sh | sh

The pins live in a PEP 735 dependency group, which uv resolves and locks. If you
must avoid uv, pip 25.1+ can read the group directly:

  python3 -m venv .venv && .venv/bin/python -m pip install --group figures
MSG
  exit 1
fi

uv sync --group figures --python "$PYTHON_VERSION"

.venv/bin/python -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL"

echo
echo "Registered kernel '$KERNEL_LABEL' from .venv"
.venv/bin/python - <<'PY'
import importlib.metadata as im
for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "plotly", "kaleido"):
    try:
        print(f"  {p:14s} {im.version(p)}")
    except im.PackageNotFoundError:
        print(f"  {p:14s} MISSING")
PY
echo
echo "Select '$KERNEL_LABEL' in Jupyter, then run notebooks/figure_{1,3,4}.ipynb"
