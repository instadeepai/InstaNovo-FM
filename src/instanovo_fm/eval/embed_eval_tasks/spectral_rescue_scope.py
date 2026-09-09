"""Scope definitions for reformulated spectral rescue evaluation.

Two evaluation scopes are supported:

``single_base`` — controlled case study
    One fixed base peptide and one fixed modified (rescue) variant, typically in
    one project.  Best for mechanistic figures, UMAP vignettes, and ablations
    where you need a clean story around a single modification.

    How to run:
        1. Set ``scope: single_base`` (default).
        2. Provide ``base_sequence`` and ``rescue_sequence`` (or rely on defaults).
        3. Build offline data: ``create_spectral_rescue_reformulated_dataset.py --scope single_base``.
        4. Eval with the single-base config; metrics describe that pair only.

``multi_base`` — benchmark panel
    Many base/modified pairs, each with an isolated query/library block.  Metrics
    are computed per pair, then aggregated (mean ± std) across the panel for
    generalization claims.

    How to run:
        1. Curate ``rescue_pairs`` (JSON manifest — see ``rescue_pairs_multi_example.json``).
        2. Set ``scope: multi_base`` and ``rescue_pairs_path`` (or inline ``rescue_pairs``).
        3. Build offline data: ``create_spectral_rescue_reformulated_dataset.py \\
           --scope multi_base --rescue_pairs_path path/to/pairs.json``.
        4. Eval with multi-base config; report ``panel_*`` aggregates plus per-pair tables.

Rows in offline parquets are tagged with ``rescue_pair_id``.  Queries from pair *A*
never retrieve library spectra from pair *B*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

RescueScope = Literal["single_base", "multi_base"]

DEFAULT_PAIR_ID = "default"

RESCUE_SCOPE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "single_base": {
        "name": "Single-base case study",
        "question": "On this one peptide pair, does embedding retrieval rescue the modified spectrum?",
        "strength": "High internal validity; ideal for figures and qualitative analysis.",
        "limit": "Does not support generalization claims across peptides or projects.",
        "dataset_builder": "create_spectral_rescue_reformulated_dataset.py --scope single_base [--profile rigorous]",
        "eval_config": "foundational_eval_spectral_rescue_reformulated_rigorous (scope defaults to single_base)",
    },
    "multi_base": {
        "name": "Multi-base benchmark panel",
        "question": "Across a curated panel of peptide pairs, how often does rescue retrieval succeed?",
        "strength": "Per-pair isolation plus panel mean/std supports broader claims.",
        "limit": "Requires enough pairs with sufficient spectra; pair curation is the main cost.",
        "dataset_builder": "create_spectral_rescue_reformulated_dataset.py --scope multi_base --rescue_pairs_path pairs.json",
        "eval_config": "foundational_eval_spectral_rescue_reformulated_multi_base",
    },
}


@dataclass(frozen=True)
class RescuePairSpec:
    """One base/modified peptide pair in a rescue experiment."""

    pair_id: str
    project_id: str
    base_sequence: str
    modified_sequence: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "project_id": self.project_id,
            "base_sequence": self.base_sequence,
            "modified_sequence": self.modified_sequence,
            "notes": self.notes,
        }


@dataclass
class RescueScopeConfig:
    scope: RescueScope
    pairs: List[RescuePairSpec] = field(default_factory=list)
    pair_key: str = "rescue_pair_id"

    @property
    def is_multi_base(self) -> bool:
        return self.scope == "multi_base"

    @property
    def is_single_base(self) -> bool:
        return self.scope == "single_base"


def _parse_pair_dict(raw: Dict[str, Any]) -> RescuePairSpec:
    missing = [key for key in ("pair_id", "project_id", "base_sequence", "modified_sequence") if key not in raw]
    if missing:
        raise ValueError(f"Rescue pair missing required keys {missing}: {raw}")
    return RescuePairSpec(
        pair_id=str(raw["pair_id"]),
        project_id=str(raw["project_id"]),
        base_sequence=str(raw["base_sequence"]),
        modified_sequence=str(raw["modified_sequence"]),
        notes=str(raw.get("notes", "")),
    )


def load_rescue_pairs_manifest(path: str | Path) -> tuple[RescueScope, List[RescuePairSpec]]:
    """Load scope and pair list from a JSON manifest."""
    payload = json.loads(Path(path).read_text())
    scope = payload.get("scope", "multi_base")
    if scope not in ("single_base", "multi_base"):
        raise ValueError(f"Invalid scope {scope!r} in {path}")
    pairs_raw = payload.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ValueError(f"Manifest {path} must contain a non-empty 'pairs' list")
    pairs = [_parse_pair_dict(item) for item in pairs_raw]
    if scope == "multi_base" and len(pairs) < 2:
        raise ValueError("multi_base scope requires at least two pairs in the manifest")
    return scope, pairs


def resolve_rescue_scope_config(
    *,
    scope: str = "single_base",
    pair_key: str = "rescue_pair_id",
    project_id: str = "PXD047134",
    base_sequence: str = "LEQGQALDDLMPAQK",
    rescue_sequence: str = "LEQGQALDDLM[UNIMOD:35]PAQK",
    rescue_pairs: Optional[Sequence[Dict[str, Any]]] = None,
    rescue_pairs_path: Optional[str] = None,
) -> RescueScopeConfig:
    """Resolve scope settings from task/dataset kwargs."""
    if scope not in ("single_base", "multi_base"):
        raise ValueError(f"Unknown rescue scope {scope!r}; expected 'single_base' or 'multi_base'")

    if rescue_pairs_path:
        manifest_scope, pairs = load_rescue_pairs_manifest(rescue_pairs_path)
        if manifest_scope != scope:
            raise ValueError(
                f"Config scope={scope!r} disagrees with manifest scope={manifest_scope!r} in {rescue_pairs_path}"
            )
        return RescueScopeConfig(scope=scope, pairs=pairs, pair_key=pair_key)

    if rescue_pairs:
        pairs = [_parse_pair_dict(dict(item)) for item in rescue_pairs]
    elif scope == "single_base":
        pairs = [
            RescuePairSpec(
                pair_id=DEFAULT_PAIR_ID,
                project_id=project_id,
                base_sequence=base_sequence,
                modified_sequence=rescue_sequence,
            )
        ]
    else:
        raise ValueError(
            "multi_base scope requires rescue_pairs or rescue_pairs_path "
            "(see instanovo/configs/rescue_pairs_multi_example.json)"
        )

    if scope == "single_base":
        if len(pairs) != 1:
            raise ValueError("single_base scope accepts exactly one pair definition")
        return RescueScopeConfig(scope=scope, pairs=pairs, pair_key=pair_key)

    if len(pairs) < 2:
        raise ValueError("multi_base scope requires at least two pairs")
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"Duplicate pair_id values in rescue panel: {pair_ids}")
    return RescueScopeConfig(scope=scope, pairs=pairs, pair_key=pair_key)


def scope_guidance_text(scope: Optional[RescueScope] = None) -> str:
    """Return human-readable guidance for one or all scopes."""
    scopes = [scope] if scope else ["single_base", "multi_base"]
    lines = []
    for name in scopes:
        info = RESCUE_SCOPE_DEFINITIONS[name]
        lines.append(f"[{name}] {info['name']}")
        lines.append(f"  Question: {info['question']}")
        lines.append(f"  Strength: {info['strength']}")
        lines.append(f"  Limit: {info['limit']}")
        lines.append(f"  Dataset: {info['dataset_builder']}")
        lines.append(f"  Eval: {info['eval_config']}")
    return "\n".join(lines)
