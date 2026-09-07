"""Guard: fetching the peptide registry must not download the whole dataset repo.

``load_peptide_registry`` in ``scripts/splitting/split_labelled_data.py`` reads one
file, ``peptide_registry.parquet``, from the HuggingFace dataset repo. It used to do
that with ``snapshot_download``, which fetches the *entire* repository. That was
harmless while the repo held only the registry; once the corpus is published it means
downloading roughly a terabyte to read about sixty megabytes, triggered by a user who
asked for neither.

These tests are static -- they parse the source rather than importing it -- so they
need no network, no token and no HuggingFace state, and they fail on the change that
would reintroduce the problem rather than on its consequences.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "splitting" / "split_labelled_data.py"


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text())


def _called_names(tree: ast.Module) -> set[str]:
    """Every function name called anywhere in the module."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def test_snapshot_download_is_not_imported() -> None:
    """The whole-repo downloader should not even be in scope."""
    imported = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "snapshot_download" not in imported, (
        "snapshot_download downloads the entire dataset repo. Use hf_hub_download with "
        "filename=REGISTRY_FILENAME to fetch only the registry."
    )


def test_snapshot_download_is_not_called() -> None:
    """Belt and braces: not reachable via an aliased or attribute call either."""
    assert "snapshot_download" not in _called_names(_tree())


def test_registry_is_fetched_with_hf_hub_download() -> None:
    """The single-file downloader is used, and told which single file."""
    tree = _tree()
    assert "hf_hub_download" in _called_names(tree)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hf_hub_download"
        ):
            kwargs = {kw.arg for kw in node.keywords}
            assert (
                "filename" in kwargs
            ), "hf_hub_download without filename= does not restrict what is fetched"
            filename = next(kw.value for kw in node.keywords if kw.arg == "filename")
            # Must be the module constant, not a literal that can drift from it.
            assert (
                isinstance(filename, ast.Name) and filename.id == "REGISTRY_FILENAME"
            ), "pass filename=REGISTRY_FILENAME so the constant stays the single source"
            return
    raise AssertionError("no direct hf_hub_download(...) call found")
