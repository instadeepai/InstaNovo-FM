"""Tests for spectral rescue scope configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_scope import (
    resolve_rescue_scope_config,
    scope_guidance_text,
)


def test_resolve_single_base_defaults():
    cfg = resolve_rescue_scope_config(scope="single_base")
    assert cfg.is_single_base
    assert len(cfg.pairs) == 1
    assert cfg.pairs[0].base_sequence == "LEQGQALDDLMPAQK"


def test_resolve_multi_base_requires_pairs():
    with pytest.raises(ValueError, match="multi_base scope requires"):
        resolve_rescue_scope_config(scope="multi_base")


def test_load_pairs_manifest(tmp_path: Path):
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "scope": "multi_base",
                "pairs": [
                    {
                        "pair_id": "a",
                        "project_id": "PXD1",
                        "base_sequence": "AAAAA",
                        "modified_sequence": "AAA[UNIMOD:35]A",
                    },
                    {
                        "pair_id": "b",
                        "project_id": "PXD1",
                        "base_sequence": "BBBBB",
                        "modified_sequence": "BBB[UNIMOD:35]B",
                    },
                ],
            }
        )
    )
    cfg = resolve_rescue_scope_config(scope="multi_base", rescue_pairs_path=str(manifest))
    assert cfg.is_multi_base
    assert [pair.pair_id for pair in cfg.pairs] == ["a", "b"]


def test_scope_guidance_mentions_both_modes():
    text = scope_guidance_text()
    assert "single_base" in text
    assert "multi_base" in text


def _make_multi_pair_fixture():
    sequences = np.array(
        [
            "PEPA",
            "PEPA",
            "PEPA",
            "PEPA[UNIMOD:35]",
            "ZZZZZ",
            "PEPB",
            "PEPB",
            "PEPB",
            "PEPB[UNIMOD:35]",
            "YYYYY",
        ],
        dtype=object,
    )
    unmodified = np.array(
        [
            "PEPA",
            "PEPA",
            "PEPA",
            "PEPA",
            "ZZZZZ",
            "PEPB",
            "PEPB",
            "PEPB",
            "PEPB",
            "YYYYY",
        ],
        dtype=object,
    )
    roles = np.array(
        [
            "reference_query",
            "positive_library",
            "positive_library",
            "modified_query",
            "negative_library",
            "reference_query",
            "positive_library",
            "positive_library",
            "modified_query",
            "negative_library",
        ],
        dtype=object,
    )
    pair_ids = np.array(["pair_a"] * 5 + ["pair_b"] * 5, dtype=object)
    meta = {
        "peptides": sequences,
        "unmodified_peptide": unmodified,
        "rescue_role": roles,
        "rescue_pair_id": pair_ids,
        "search_project": np.array(["PXD1"] * 10, dtype=object),
        "usi": np.array([f"mzspec:PXD1:run:scan:{i}" for i in range(10)], dtype=object),
    }
    emb = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    return emb, meta


def test_multi_base_runs_isolated_pair_metrics():
    emb, meta = _make_multi_pair_fixture()
    task = SpectralRescueTaskReformulated(
        scope="multi_base",
        rescue_pairs=[
            {
                "pair_id": "pair_a",
                "project_id": "PXD1",
                "base_sequence": "PEPA",
                "modified_sequence": "PEPA[UNIMOD:35]",
            },
            {
                "pair_id": "pair_b",
                "project_id": "PXD1",
                "base_sequence": "PEPB",
                "modified_sequence": "PEPB[UNIMOD:35]",
            },
        ],
        k_values=[1, 2],
        save_artifacts=False,
        save_plots=False,
    )

    results = task.run(emb, meta)

    assert "error" not in results
    assert results["scope"] == "multi_base"
    assert results["num_pairs"] == 2
    assert "panel_modified_query_recall@1_mean" in results
    assert set(results["pair_results"]) == {"pair_a", "pair_b"}
    assert results["pair_results"]["pair_a"]["num_queries"] == 2
    assert results["pair_results"]["pair_b"]["num_queries"] == 2


def test_single_base_rejects_multiple_pair_ids():
    emb, meta = _make_multi_pair_fixture()
    task = SpectralRescueTaskReformulated(scope="single_base", save_artifacts=False, save_plots=False)
    results = task.run(emb, meta)
    assert "error" in results
    assert "single_base" in results["error"]
