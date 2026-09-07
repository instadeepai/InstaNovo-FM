"""Pin the upload call against the huggingface_hub API it actually calls.

Two uploads have now been lost to a signature mismatch discovered on the cluster
rather than locally: ``upload_large_folder`` takes neither ``path_in_repo`` nor
``token``, and each cost a submit-and-wait cycle to find out. The call is made through
``getattr``, so nothing else in the codebase would catch it.

These tests read the installed signature rather than hard-coding a parameter list, so
they keep working across upgrades and fail if a future version removes something the
call depends on.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")

UPLOAD = Path(__file__).resolve().parents[1] / "scripts" / "release" / "hf_upload.py"


def _uploader_kwargs() -> set[str]:
    """The keyword names passed to the uploader call in hf_upload.py."""
    tree = ast.parse(UPLOAD.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "uploader"
        ):
            return {k.arg for k in node.keywords if k.arg}
    raise AssertionError("no uploader(...) call found in hf_upload.py")


def test_every_kwarg_is_accepted_by_upload_large_folder() -> None:
    """The call must not pass anything the installed API rejects."""
    accepted = set(inspect.signature(huggingface_hub.HfApi.upload_large_folder).parameters)
    rejected = _uploader_kwargs() - accepted
    assert not rejected, (
        f"upload_large_folder in huggingface_hub {huggingface_hub.__version__} does not "
        f"accept {sorted(rejected)}. This fails only at upload time, on the cluster."
    )


def test_the_call_still_passes_what_it_needs() -> None:
    """Guard against a fix that silently drops the arguments doing the work."""
    passed = _uploader_kwargs()
    for required in ("repo_id", "repo_type", "folder_path", "allow_patterns"):
        assert required in passed, f"{required} is not passed; the upload would misplace files"


def test_path_in_repo_is_not_used() -> None:
    """It does not exist on this API, and reaching for it means the layout is wrong.

    Files land where the local tree puts them relative to folder_path, which is why the
    staging tree mirrors the repository and the unit is selected with allow_patterns.
    """
    assert "path_in_repo" not in _uploader_kwargs()


def test_token_is_on_the_client_not_the_call() -> None:
    """upload_large_folder takes no token, so HfApi must carry it."""
    src = UPLOAD.read_text()
    assert "HfApi(token=token)" in src, "HfApi built without a token: the upload would 401"
    assert "token" not in _uploader_kwargs()
