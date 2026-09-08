"""Smoke tests for every Typer CLI under :mod:`scripts`."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _typer_modules() -> list[str]:
    """Discover modules that assign a ``typer.Typer`` application."""
    modules = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "typer"
            and node.func.attr == "Typer"
            for node in ast.walk(tree)
        ):
            relative = path.relative_to(SCRIPTS_DIR.parent).with_suffix("")
            modules.append(".".join(relative.parts))
    return sorted(modules)


@pytest.mark.parametrize("module_name", _typer_modules())
def test_cli_help(module_name: str) -> None:
    """Every script CLI should expose usable help."""
    app = importlib.import_module(module_name).app
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, (
        f"help failed for {module_name}: exit={result.exit_code} "
        f"exception={result.exception!r}"
    )
    assert "Usage:" in result.output
