"""The wheel has to build, and has to carry the package's data files.

Neither was true before this test existed. ``[tool.hatch.build.targets.wheel]``
listed ``packages = ["src/instanovo_fm"]`` *and* force-included ``configs/``,
which adds every config twice and makes hatchling refuse:

    ValueError: A second file is being added to the wheel archive at the same
    path: `instanovo_fm/configs/denovo.yaml`

An editable install never exercises that path, so the failure only showed up on
``uv build`` -- which is what a publish workflow does. The data files still have
to be present, since ``from_pretrained`` reads ``models.json`` and the CLI
composes the Hydra configs through ``importlib.resources``.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# One entry per thing the package needs at runtime but does not import.
REQUIRED = (
    "instanovo_fm/models.json",
    "instanovo_fm/configs/foundational.yaml",
    "instanovo_fm/configs/denovo.yaml",
    "instanovo_fm/configs/model/foundation_base.yaml",
    "instanovo_fm/configs/residues/default.yaml",
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    """Build a wheel into a temporary directory and open it."""
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"uv build --wheel failed:\n{result.stdout}\n{result.stderr}")

    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {[w.name for w in wheels]}"
    return zipfile.ZipFile(wheels[0])


@pytest.mark.slow
def test_wheel_carries_the_data_files(wheel: zipfile.ZipFile) -> None:
    """A pip install has to be usable, so the configs and the registry must ship."""
    names = set(wheel.namelist())
    missing = [path for path in REQUIRED if path not in names]
    assert not missing, f"absent from the wheel: {missing}"


@pytest.mark.slow
def test_wheel_has_no_duplicate_entries(wheel: zipfile.ZipFile) -> None:
    """Duplicates are what force-include caused, and they fail the build outright."""
    names = wheel.namelist()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"added to the wheel more than once: {duplicates}"


@pytest.mark.slow
def test_wheel_ships_every_config(wheel: zipfile.ZipFile) -> None:
    """Every config in the tree, not just the ones spot-checked above."""
    on_disk = {
        f"instanovo_fm/configs/{path.relative_to(REPO / 'src/instanovo_fm/configs').as_posix()}"
        for path in (REPO / "src/instanovo_fm/configs").rglob("*.yaml")
    }
    in_wheel = set(wheel.namelist())
    assert on_disk <= in_wheel, f"configs left out of the wheel: {sorted(on_disk - in_wheel)}"
