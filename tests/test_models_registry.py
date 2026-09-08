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


def test_version_is_not_written_twice() -> None:
    """pyproject.toml is the only place a version is declared.

    ``__version__`` was a second literal in ``__init__.py`` and the two had
    already drifted: pyproject said 0.1.0 while the package still said
    0.1.0.dev0.
    """
    import importlib.metadata

    import instanovo_fm

    assert instanovo_fm.__version__ == importlib.metadata.version("instanovo-fm")


def test_asset_urls_name_the_tag_for_this_version(registry: dict[str, Any]) -> None:
    """The release tag in every asset URL has to match the version being released.

    A download is an exact URL, so bumping the version without moving the URLs
    leaves every id resolving to a 404 on a tag that will never exist.
    """
    import instanovo_fm

    expected = f"{RELEASE_PREFIX}v{instanovo_fm.__version__}/"
    for models in registry.values():
        for model_id, info in models.items():
            assert info["remote"].startswith(expected), (
                f"{model_id} points at {info['remote']}, but this is version "
                f"{instanovo_fm.__version__}, so the tag should be v{instanovo_fm.__version__}"
            )


def test_describe_pretrained_covers_every_id(registry: dict[str, Any]) -> None:
    """Describing everything must agree with listing everything."""
    described = FoundationModel.describe_pretrained()
    assert list(described) == FoundationModel.get_pretrained()
    assert described == registry[FOUNDATION_MODEL_TYPE]
    assert DownstreamDeNovo.describe_pretrained() == registry[DENOVO_MODEL_TYPE]


def test_describe_pretrained_returns_one_entry(registry: dict[str, Any]) -> None:
    """A single id returns just that entry."""
    entry = FoundationModel.describe_pretrained("instanovo-fm-v0.1.0")
    assert entry == registry[FOUNDATION_MODEL_TYPE]["instanovo-fm-v0.1.0"]


def test_describe_pretrained_rejects_an_unknown_id() -> None:
    """And says what the options are, as from_pretrained does."""
    with pytest.raises(ValueError, match="not found in models.json"):
        FoundationModel.describe_pretrained("no-such-model")


def test_describe_pretrained_hands_back_a_copy() -> None:
    """A caller poking at the result must not corrupt the registry for the next one."""
    first = FoundationModel.describe_pretrained("instanovo-fm-v0.1.0")
    first["layers"] = 999
    assert FoundationModel.describe_pretrained("instanovo-fm-v0.1.0")["layers"] == 12

    everything = FoundationModel.describe_pretrained()
    everything["instanovo-fm-v0.1.0"]["corpus"] = "nonsense"
    assert FoundationModel.describe_pretrained()["instanovo-fm-v0.1.0"]["corpus"] == "LCFM"


# What a caller needs in order to choose, per family. Kept separate on purpose:
# `masking` and `pairwise_bias` describe how a foundation model was pretrained and
# say nothing about a de novo model trained from scratch, so requiring them
# everywhere would only invite filling them in with something untrue.
REQUIRED_FIELDS = {
    "": ("description", "corpus", "training_steps"),
    "foundational": ("masking", "pairwise_bias", "layers"),
    "downstream_denovo": ("encoder_init",),
}


def test_every_entry_describes_itself(registry: dict[str, Any]) -> None:
    """The fields a caller chooses between checkpoints on must be present."""
    for model_type, models in registry.items():
        required = REQUIRED_FIELDS[""] + REQUIRED_FIELDS.get(model_type, ())
        for model_id, info in models.items():
            missing = [field for field in required if field not in info]
            assert not missing, f"{model_type}/{model_id} does not record {missing}"


def test_every_family_has_its_required_fields_declared(registry: dict[str, Any]) -> None:
    """A new model type must say what describes it, rather than inheriting nothing."""
    undeclared = [t for t in registry if t not in REQUIRED_FIELDS]
    assert not undeclared, f"REQUIRED_FIELDS says nothing about {undeclared}"


def test_the_de_novo_variants_cover_the_three_encoder_treatments(registry: dict[str, Any]) -> None:
    """The paper trains the de novo model three ways: fine-tuned, frozen, from scratch."""
    models = registry["downstream_denovo"]
    assert {info["encoder_init"] for info in models.values()} == {
        "fine-tuned",
        "frozen",
        "from scratch",
    }
    # All three share the schedule stated in Methods.
    for model_id, info in models.items():
        assert info["training_steps"] == 2500000, model_id
        assert info["warmup_steps"] == 100000, model_id
        assert info["batch_size"] == 128, model_id


def test_the_published_de_novo_model_is_the_fine_tuned_one(registry: dict[str, Any]) -> None:
    """It is the variant benchmarked against IN v1.2, Casanovo and XuanjiNovo."""
    published = registry["downstream_denovo"]["instanovo-fm-denovo-v0.1.0"]
    assert published["encoder_init"] == "fine-tuned"
    assert published["encoder_unfrozen_at_step"] == 100000


def test_the_published_model_matches_the_paper(registry: dict[str, Any]) -> None:
    """Metadata nothing reads is metadata that rots, so pin it to the manuscript.

    Methods states the deployed model: model dimension 768, 12 attention heads,
    12 layers, feedforward 3072, about 89.5M parameters, ~230,000 steps on LCFM
    with Thompson-span masking and no pairwise attention bias.
    """
    published = registry[FOUNDATION_MODEL_TYPE]["instanovo-fm-v0.1.0"]
    assert published["layers"] == 12
    assert published["model_dimension"] == 768
    assert published["attention_heads"] == 12
    assert published["feedforward_dimension"] == 3072
    assert published["parameters"] == "89.5M"
    assert published["training_steps"] == 230000
    assert published["corpus"] == "LCFM"
    assert published["masking"].startswith("thompson_span")
    assert published["pairwise_bias"] is False


def test_the_factorial_covers_all_four_cells(registry: dict[str, Any]) -> None:
    """Two axes, masking strategy and pairwise bias, so four LCFM checkpoints."""
    lcfm = {
        (info["masking"].split("_")[0], info["pairwise_bias"])
        for info in registry[FOUNDATION_MODEL_TYPE].values()
        if info["corpus"] == "LCFM"
    }
    assert lcfm == {("thompson", False), ("thompson", True), ("signal", False), ("signal", True)}
