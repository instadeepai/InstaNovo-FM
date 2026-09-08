"""The pretrained-checkpoint registry, and the wiring that reads it.

``from_pretrained`` used to read ``models.json`` out of the installed
``instanovo`` package, whose registry has only ``transformer`` and ``diffusion``
keys, so every by-id lookup here found nothing. These tests hold the fixed
wiring in place: the registry ships with *this* package, the classes look for
their own key in it, and nothing about the cluster the checkpoints were trained
on reaches the published file.
"""

from __future__ import annotations

import importlib.resources as resources
import json
import re
from typing import Any

import pytest

from instanovo_fm.downstream.de_novo_sequencing.model import MODEL_TYPE as DENOVO_MODEL_TYPE
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.model.encoder import MODEL_TYPE as FOUNDATION_MODEL_TYPE
from instanovo_fm.model.encoder import FoundationModel

RELEASE_PREFIX = "https://github.com/instadeepai/InstaNovo-FM/releases/download/"


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """The registry as the installed package exposes it."""
    text = resources.files("instanovo_fm").joinpath("models.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_registry_ships_with_the_package(registry: dict[str, Any]) -> None:
    """It has to be readable through importlib.resources, not just present in the repo."""
    assert registry, "models.json resolved but is empty"


def test_model_types_have_a_registry_key(registry: dict[str, Any]) -> None:
    """Each class looks up its own MODEL_TYPE, so a typo there silently finds nothing.

    This is the bug the wiring had: the downstream model's MODEL_TYPE was
    "transformer", so it offered InstaNovo's checkpoints for an architecture its
    own ``load`` does not build.
    """
    for model_type in (FOUNDATION_MODEL_TYPE, DENOVO_MODEL_TYPE):
        assert model_type in registry, f"MODEL_TYPE {model_type!r} has no key in models.json"


def test_foundation_models_are_listed(registry: dict[str, Any]) -> None:
    """The paper's checkpoints are the point of the registry."""
    ids = registry[FOUNDATION_MODEL_TYPE]
    assert "instanovo-fm-v0.1.0" in ids, "the published model must be registered"
    assert len(ids) >= 5, "the published model plus the factorial and MCFM comparisons"


def test_every_entry_has_a_remote(registry: dict[str, Any]) -> None:
    """``remote`` is the only field ``from_pretrained`` reads."""
    for model_type, models in registry.items():
        for model_id, info in models.items():
            assert "remote" in info, f"{model_type}/{model_id} has no 'remote'"
            assert info["remote"].endswith(".ckpt"), f"{model_type}/{model_id} is not a .ckpt"


def test_remotes_are_release_assets_of_this_repository(registry: dict[str, Any]) -> None:
    """Checkpoints are published as release assets, the way InstaNovo publishes its own."""
    for models in registry.values():
        for model_id, info in models.items():
            assert info["remote"].startswith(
                RELEASE_PREFIX
            ), f"{model_id} does not point at a release asset of this repository: {info['remote']}"


def test_registry_names_no_internal_infrastructure() -> None:
    """The checkpoints were trained on a private cluster; the registry must not say so.

    The training runs live in an object-store bucket under per-experiment UUIDs.
    Those are exactly what the CI leak gate rejects, and a registry is a tempting
    place to paste them, so this fails before the gate has to.
    """
    text = resources.files("instanovo_fm").joinpath("models.json").read_text(encoding="utf-8")
    forbidden = {
        "an object-store URI": re.compile(r"s3://", re.I),
        "a cluster bucket name": re.compile(r"[a-z0-9-]*denovo-s-[0-9a-f]{8,}", re.I),
        "an experiment UUID": re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
        ),
    }
    for what, pattern in forbidden.items():
        match = pattern.search(text)
        assert match is None, f"models.json contains {what}: {match.group(0)!r}"


def test_get_pretrained_reads_this_registry(registry: dict[str, Any]) -> None:
    """``get_pretrained`` and ``from_pretrained`` must agree on where to look."""
    assert FoundationModel.get_pretrained() == list(registry[FOUNDATION_MODEL_TYPE])
    assert DownstreamDeNovo.get_pretrained() == list(registry[DENOVO_MODEL_TYPE])


def test_unknown_id_lists_what_is_available() -> None:
    """A wrong id should say what the options are rather than fail obscurely."""
    with pytest.raises(ValueError, match="not found in models.json"):
        FoundationModel.from_pretrained("no-such-model")


def test_a_path_that_does_not_exist_is_reported_as_such(tmp_path: Any) -> None:
    """The local-path branch is the one that works before a release is cut."""
    missing = tmp_path / "model_best.ckpt"
    with pytest.raises(FileNotFoundError):
        FoundationModel.from_pretrained(str(missing))
