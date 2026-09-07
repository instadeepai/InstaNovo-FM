"""The external-baseline adapters must not pull their dependencies at import time.

Casanovo pins numpy<2.0 while instanovo requires numpy>=2.0.2, so Casanovo can
never be installed alongside InstaNovo-FM. XuanjiNovo needs a CUDA 12.1 image
with ctcdecode, imputer-pytorch and cupy. Both therefore run from their own
images (see docker/README.md).

That arrangement only works while every third-party baseline import sits inside a
function. If one drifts to module scope, importing instanovo_fm.eval starts
requiring a package that cannot be installed, and the whole eval harness breaks
in the default environment. These tests pin the property down.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

EVAL_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "instanovo_fm" / "eval"

# Packages that are absent from the default environment by design.
ISOLATED = {"casanovo", "massnet", "ctcdecode", "imputer", "cupy", "cuml", "glass_box_umap"}


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names only; anything nested in a function is fine."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:  # body == module scope
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # a guarded try/except ImportError at module scope is acceptable
            continue
    return found


@pytest.mark.parametrize("path", sorted(EVAL_DIR.glob("*.py")), ids=lambda p: p.name)
def test_isolated_deps_are_not_imported_at_module_scope(path: pathlib.Path) -> None:
    offenders = _module_level_imports(path) & ISOLATED
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)} at module scope. These packages "
        "cannot be installed in the default environment (see docker/README.md); "
        "move the import inside the function that needs it."
    )


def test_eval_package_imports_without_baseline_deps() -> None:
    """Importing the eval package must not require any isolated dependency."""
    import importlib

    importlib.import_module("instanovo_fm.eval")

    import sys

    leaked = ISOLATED & set(sys.modules)
    assert not leaked, f"importing instanovo_fm.eval pulled in {sorted(leaked)}"
