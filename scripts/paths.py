"""Canonical paths to committed data and asset files used by pipeline CLIs.

Defaults resolve from the repository root so Typer options work regardless of
the caller's current working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEARCH_DATA = REPO_ROOT / "data" / "search_data.xlsx"
DEFAULT_GOLD_STANDARD_MODS = (
    REPO_ROOT / "assets" / "mod_dicts" / "gold_standard_modifications.xlsx"
)
DEFAULT_AMBIGUOUS_MODS = (
    REPO_ROOT / "assets" / "mod_dicts" / "PXD009449_ambiguous_mods.xlsx"
)
DEFAULT_RESIDUE_MASSES = REPO_ROOT / "assets" / "mod_dicts" / "residue_masses.yaml"
DEFAULT_MOD_DICTS_DIR = REPO_ROOT / "assets" / "mod_dicts"
DEFAULT_TMT_PROJECTS_YAML = REPO_ROOT / "assets" / "bad_tmt_projects.yaml"
