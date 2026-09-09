# Runs classification

"""Run-level embedding evaluation for runs_classification_dataset.

Each ``.ipc`` file is one LC-MS **run**.  The core output is a mapping::

    run_key = (condition_from_filename, project_name)  →  embedding (768-d)

A project folder (PXD*) contains many runs; each run gets its own embedding.
Clustering / UMAP operates on **runs within each project** (one plot per PXD*),
coloured by experimental condition from ``manifest.yaml``.

Detailed run conditions are stored separately in ``run_conditions.json`` for
downstream analysis after the clustering step.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import Dataset as HFDataset
from datasets import Value
from omegaconf import OmegaConf
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.eval import embedding_io
from instanovo_fm.model import FoundationModel
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame
from instanovo.utils.residues import ResidueSet
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

DEFAULT_CHECKPOINT = (
    "s3://<outputs-bucket>/output/<run-id>/checkpoints/instanovo-foundational-base/model_best.ckpt"
)
DEFAULT_DATASET_ROOT = "<data-root>/miscellaneous/runs_classification_dataset"

IPC_COLUMN_REMAP = {
    "mz": "mz_array",
    "intensity": "intensity_array",
    "rt": "retention_time",
}

# Parsed from filename for extra condition detail (stored in run_conditions.json).
_FILENAME_FRAG_PATTERNS = [
    (re.compile(r"ETciD|EThcD", re.I), "ETD"),
    (re.compile(r"UVPD", re.I), "UVPD"),
    (re.compile(r"\bECD\b", re.I), "ECD"),
    (re.compile(r"\bEID\b", re.I), "EID"),
    (re.compile(r"\bHCD\b", re.I), "HCD"),
    (re.compile(r"\bCID\b", re.I), "CID"),
]
_FRAC_PATTERN = re.compile(r"frac(\d+)", re.I)


@dataclass(frozen=True)
class RunKey:
    """Unique run identity: (condition from filename, project name)."""

    condition: str
    project: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.condition, self.project)

    def as_str(self) -> str:
        return f"{self.condition}::{self.project}"


@dataclass
class RunRecord:
    """One IPC file = one run."""

    ipc_path: Path
    run_key: RunKey
    source_file: str
    conditions: Dict[str, Any] = field(default_factory=dict)


def _ipc_stem(ipc_path: Path) -> str:
    name = ipc_path.name
    for suffix in (".mzML.ipc", ".ipc"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ipc_path.stem


def _parse_filename_conditions(stem: str) -> Dict[str, Any]:
    """Extract condition hints from the run filename (no manifest)."""
    detail: Dict[str, Any] = {"condition_from_filename": stem}

    for pattern, label in _FILENAME_FRAG_PATTERNS:
        if pattern.search(stem):
            detail["fragmentation_hint"] = label
            break

    frac = _FRAC_PATTERN.search(stem)
    if frac:
        detail["fraction"] = int(frac.group(1))

    return detail


def discover_runs(dataset_root: Path, project: Optional[str] = None) -> List[RunRecord]:
    """Find run IPC files under ``PXD*/`` project folders."""
    runs: List[RunRecord] = []
    pattern = f"{project}/*.ipc" if project else "PXD*/*.ipc"
    for ipc_path in sorted(dataset_root.glob(pattern)):
        if ".cache" in ipc_path.parts:
            continue
        proj = ipc_path.parent.name
        if not proj.startswith("PXD"):
            continue
        stem = _ipc_stem(ipc_path)
        run_key = RunKey(condition=stem, project=proj)
        runs.append(
            RunRecord(
                ipc_path=ipc_path,
                run_key=run_key,
                source_file=str(ipc_path.relative_to(dataset_root)),
                conditions=_parse_filename_conditions(stem),
            )
        )
    return runs


def summarize_ipc_conditions(ipc_path: Path) -> Dict[str, Any]:
    """Per-run acquisition stats from IPC columns."""
    cols = ["frag_type", "collision_energy", "retention_time", "precursor_charge"]
    available = [c for c in cols if c in pl.read_ipc_schema(ipc_path)]
    if not available:
        return {"n_spectra": pl.scan_ipc(ipc_path).select(pl.len()).collect().item()}

    df = pl.read_ipc(ipc_path, columns=available)
    stats: Dict[str, Any] = {"n_spectra": len(df)}

    if "frag_type" in df.columns:
        counts = df.filter(pl.col("frag_type").is_not_null()).group_by("frag_type").len().sort("len", descending=True)
        if len(counts):
            stats["dominant_frag_type"] = counts["frag_type"][0]
            stats["frag_type_counts"] = {row["frag_type"]: row["len"] for row in counts.iter_rows(named=True)}

    if "collision_energy" in df.columns:
        ce = df["collision_energy"].drop_nulls()
        if len(ce):
            stats["mean_collision_energy"] = float(ce.mean())
            stats["median_collision_energy"] = float(ce.median())

    if "retention_time" in df.columns:
        rt = df["retention_time"].drop_nulls()
        if len(rt):
            stats["rt_min"] = float(rt.min())
            stats["rt_max"] = float(rt.max())

    if "precursor_charge" in df.columns:
        ch = df["precursor_charge"].drop_nulls()
        if len(ch):
            stats["mean_precursor_charge"] = float(ch.mean())

    return stats


def _resolve_checkpoint(path: str, cache_dir: Path) -> str:
    if not path.startswith("s3://"):
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt = cache_dir / Path(path).name
    if local_ckpt.exists() and local_ckpt.stat().st_size > 0:
        logger.info(f"Using cached checkpoint at {local_ckpt}")
        return str(local_ckpt)

    logger.info(f"Downloading checkpoint from {path}")
    S3FileHandler().download(path, str(local_ckpt))
    return str(local_ckpt)


def _load_model_and_processor(
    checkpoint_path: str,
    residues_config: Path,
    device: torch.device,
    cache_dir: Path,
) -> tuple[FoundationModel, FoundationalDataProcessor, Any]:
    local_ckpt = _resolve_checkpoint(checkpoint_path, cache_dir)
    model, model_cfg = FoundationModel.load(local_ckpt)
    model = model.to(device).eval()

    res_cfg = OmegaConf.load(residues_config)
    residue_set = ResidueSet(residue_masses=res_cfg.residues)

    masking_cfg = model_cfg.get("masking", {})
    processor = FoundationalDataProcessor(
        n_peaks=model_cfg.get("n_peaks", 200),
        min_mz=model_cfg.get("min_mz", 50.0),
        max_mz=model_cfg.get("max_mz", 2500.0),
        min_intensity=model_cfg.get("min_intensity", 0.01),
        use_spectrum_utils=model_cfg.get("use_spectrum_utils", False),
        normalize_mz=model_cfg.get("normalize_mz", True),
        peak_ordering=masking_cfg.get("ordering_strategy", model_cfg.get("peak_ordering", "sorted")),
        residue_set=residue_set,
        annotated=False,
        masking_strategy="none",
    )
    processor._keep_non_tensor_metadata = True
    return model, processor, model_cfg


def _build_run_dataloader(
    ipc_path: Path,
    processor: FoundationalDataProcessor,
    batch_size: int,
    max_spectra: Optional[int],
    seed: int,
) -> torch.utils.data.DataLoader:
    sdf = SpectrumDataFrame.load(
        source=str(ipc_path),
        lazy=False,
        is_annotated=False,
        shuffle=False,
        add_source_file_column=True,
        column_mapping=IPC_COLUMN_REMAP,
    )
    df = sdf.collect_chunked(max_samples=max_spectra, seed=seed)
    hf = HFDataset.from_pandas(df.to_pandas())
    hf = hf.add_column("prediction_id", np.arange(len(hf)), feature=Value("int32"))
    meta_cols = ["prediction_id"]
    for col in (
        "scan",
        "index",
        "header",
        "source_file",
        "rt",
        "retention_time",
        "precursor_charge",
        "precursor_mz",
        "precursor_intensity",
        "frag_type",
        "collision_energy",
        "scale_factor",
        "isolation_target",
        "lower_offset",
        "upper_offset",
    ):
        if col in hf.column_names:
            meta_cols.append(col)
    processor.add_metadata_columns(meta_cols)
    dataset = processor.process_dataset(hf, return_format="torch")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=processor.collate_fn,
    )


RUN_POOLING_CHOICES = ("mean", "weighted_mean", "topk_concat")


def _needs_spectrum_confidence(
    run_pooling: str,
    confidence_top_k: Optional[int],
) -> bool:
    return run_pooling in ("weighted_mean", "topk_concat") or confidence_top_k is not None


def _select_top_k_by_confidence(
    spectrum_embeddings: np.ndarray,
    spectrum_confidence: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-k spectra ranked by per-spectrum confidence."""
    if top_k <= 0:
        raise ValueError(f"confidence_top_k must be positive, got {top_k}")
    k = min(top_k, len(spectrum_embeddings))
    order = np.argsort(spectrum_confidence)[-k:][::-1]
    return spectrum_embeddings[order], spectrum_confidence[order]


def pool_spectra_to_run_embedding(
    spectrum_embeddings: np.ndarray,
    run_pooling: str = "mean",
    *,
    spectrum_confidence: Optional[np.ndarray] = None,
    confidence_top_k: Optional[int] = None,
) -> np.ndarray:
    """Pool spectrum-level embeddings into a run-level representation.

    Spectrum embeddings from the FM are L2-normalised per spectrum. For
    ``mean`` and ``weighted_mean`` the run vector is re-normalised so cosine
    distance reflects direction. ``topk_concat`` selects the top-k spectra by
    classifier confidence and flattens their embeddings (no run-level L2).

    Args:
        spectrum_embeddings: ``(N, D)`` spectrum vectors.
        run_pooling: ``mean``, ``weighted_mean``, or ``topk_concat``.
        spectrum_confidence: ``(N,)`` scores from the classification head
            (logits → softmax, aggregated per spectrum).
        confidence_top_k: Required for ``topk_concat``. Optional for ``mean`` /
            ``weighted_mean``: keep only the top-k spectra by confidence before
            pooling (confidence filtering + aggregation).

    Returns:
        ``mean`` / ``weighted_mean``: ``(D,)`` L2-normalised vector.
        ``topk_concat``: ``(K * D,)`` flattened stack of top-k spectrum embs.
    """
    if len(spectrum_embeddings) == 0:
        raise ValueError("Cannot pool empty spectrum_embeddings")
    if run_pooling not in RUN_POOLING_CHOICES:
        raise ValueError(f"Unknown run_pooling {run_pooling!r}; expected one of {RUN_POOLING_CHOICES}")

    if run_pooling == "topk_concat":
        if confidence_top_k is None:
            raise ValueError("topk_concat requires --confidence-top-k (number of spectra to keep)")
        if spectrum_confidence is None:
            raise ValueError("topk_concat requires per-spectrum confidence scores (classification head + softmax); none were computed")
        top_embeddings, _ = _select_top_k_by_confidence(spectrum_embeddings, spectrum_confidence, confidence_top_k)
        return top_embeddings.reshape(-1).astype(np.float32)

    embeddings = spectrum_embeddings
    confidence = spectrum_confidence

    if confidence_top_k is not None:
        if confidence is None:
            raise ValueError("confidence_top_k requires per-spectrum confidence scores (classification head + softmax); none were computed")
        embeddings, confidence = _select_top_k_by_confidence(embeddings, confidence, confidence_top_k)

    if run_pooling == "mean":
        pooled = embeddings.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    if run_pooling == "weighted_mean":
        if confidence is None:
            raise ValueError("weighted_mean requires per-spectrum confidence scores (classification head + softmax); none were computed")
        weights = np.clip(confidence.astype(np.float64), 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(len(weights), dtype=np.float64)
        pooled = np.average(embeddings, axis=0, weights=weights)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    raise ValueError(f"Unhandled run_pooling: {run_pooling}")


def compute_run_embedding(
    model: FoundationModel,
    processor: FoundationalDataProcessor,
    ipc_path: Path,
    device: torch.device,
    batch_size: int,
    max_spectra: Optional[int],
    embedding_pooling: str,
    run_pooling: str,
    confidence_top_k: Optional[int],
    confidence_temperature: float,
    seed: int,
    compute_confidence: bool = False,
) -> tuple[np.ndarray, int, int, np.ndarray, Dict[str, np.ndarray]]:
    """Embed every spectrum in the IPC file, then pool to a run-level vector.

    Returns:
        (run_embedding, n_spectra_embedded, n_spectra_pooled, spectrum_embeddings, metadata)
    """
    loader = _build_run_dataloader(ipc_path, processor, batch_size, max_spectra, seed)

    need_confidence = _needs_spectrum_confidence(run_pooling, confidence_top_k) or compute_confidence
    spectrum_embeddings, metadata, _faiss_index = embedding_io.generate(
        model,
        loader,
        device=device,
        embedding_pooling=embedding_pooling,
        compute_confidence=need_confidence,
        generate_theoretical=False,
        show_progress=False,
        confidence_temperature=confidence_temperature,
    )
    spectrum_confidence = metadata.get("spectrum_confidence")

    run_embedding = pool_spectra_to_run_embedding(
        spectrum_embeddings,
        run_pooling=run_pooling,
        spectrum_confidence=spectrum_confidence,
        confidence_top_k=confidence_top_k,
    )
    if run_pooling == "topk_concat" or confidence_top_k is not None:
        n_pooled = min(confidence_top_k or 0, len(spectrum_embeddings))
    else:
        n_pooled = len(spectrum_embeddings)
    return run_embedding, len(spectrum_embeddings), n_pooled, spectrum_embeddings, metadata


def build_run_to_embedding_map(
    model: FoundationModel,
    processor: FoundationalDataProcessor,
    runs: List[RunRecord],
    device: torch.device,
    batch_size: int,
    max_spectra: Optional[int],
    embedding_pooling: str,
    run_pooling: str,
    confidence_top_k: Optional[int],
    confidence_temperature: float,
    seed: int,
    save_spectrum_embeddings_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute run_key → embedding for each IPC file."""
    run_to_emb: Dict[str, Dict[str, Any]] = {}
    all_spec_embs = []
    all_spec_meta = []

    for run in tqdm(runs, desc="Computing run embeddings"):
        ipc_conditions = summarize_ipc_conditions(run.ipc_path)
        run.conditions.update(ipc_conditions)
        run.conditions["ipc_path"] = str(run.ipc_path)
        run.conditions["source_file"] = run.source_file

        embedding, n_embedded, n_pooled, spec_embs, spec_meta = compute_run_embedding(
            model,
            processor,
            run.ipc_path,
            device,
            batch_size,
            max_spectra,
            embedding_pooling,
            run_pooling,
            confidence_top_k,
            confidence_temperature,
            seed,
            compute_confidence=(save_spectrum_embeddings_dir is not None),
        )

        if save_spectrum_embeddings_dir is not None:
            all_spec_embs.append(spec_embs)
            all_spec_meta.append(spec_meta)

        key = run.run_key.as_str()
        run_to_emb[key] = {
            "run_key": list(run.run_key.as_tuple()),
            "condition": run.run_key.condition,
            "project": run.run_key.project,
            "embedding": embedding,
            "n_spectra_embedded": n_embedded,
            "n_spectra_pooled": n_pooled,
            "embedding_dim": int(embedding.shape[0]),
        }
        logger.info(f"run {run.run_key.as_tuple()}: {n_embedded:,} spectra → {n_pooled:,} pooled → run embedding ({embedding.shape[0]}-d)")

    if save_spectrum_embeddings_dir is not None and all_spec_embs:
        merged_embs = np.concatenate(all_spec_embs, axis=0)
        merged_meta = {}
        keys = all_spec_meta[0].keys()
        for k in keys:
            merged_meta[k] = np.concatenate([m[k] for m in all_spec_meta], axis=0)
        _validate_spectrum_cache_alignment(
            merged_embs,
            merged_meta,
            require_confidence=True,
            context=str(save_spectrum_embeddings_dir),
        )

        import faiss

        d = merged_embs.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(merged_embs.astype(np.float32))

        save_spectrum_embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_io.save(
            merged_embs,
            merged_meta,
            faiss_index,
            save_spectrum_embeddings_dir,
            embedding_pooling=embedding_pooling,
        )

    return run_to_emb


def pooling_artifact_suffix(
    spectrum_pooling: str,
    run_pooling: str = "mean",
    confidence_top_k: Optional[int] = None,
) -> str:
    """Filename tag encoding pooling stages, e.g. ``spec-mean_pool__run-topk_concat__k-256``."""
    suffix = f"spec-{spectrum_pooling}__run-{run_pooling}"
    if confidence_top_k is not None:
        suffix += f"__k-{confidence_top_k}"
    return suffix


def _sanitize_batch_tag(batch_tag: Optional[str]) -> Optional[str]:
    if not batch_tag:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", batch_tag.strip())
    return cleaned or None


def build_project_output_dir_name(
    project: str,
    spectrum_pooling: str,
    run_pooling: str,
    confidence_top_k: Optional[int] = None,
    timestamp: Optional[str] = None,
    batch_tag: Optional[str] = None,
) -> str:
    """Relative path string for a project result tree (legacy helper)."""
    pooling_tag = pooling_artifact_suffix(spectrum_pooling, run_pooling, confidence_top_k)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_batch_tag(batch_tag) or "default"
    return f"{tag}/{project}/{pooling_tag}/{ts}"


def resolve_project_output_dir(
    output_root: Path,
    project: str,
    spectrum_pooling: str,
    run_pooling: str,
    confidence_top_k: Optional[int] = None,
    timestamp: Optional[str] = None,
    batch_tag: Optional[str] = None,
) -> Path:
    """Create and return ``{batch}/{project}/{pooling}/{timestamp}/`` with subdirs."""
    rel = build_project_output_dir_name(
        project,
        spectrum_pooling,
        run_pooling,
        confidence_top_k,
        timestamp=timestamp,
        batch_tag=batch_tag,
    )
    out = output_root / rel
    for sub in ("figures", "coords", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def _write_pooling_run_index(
    output_root: Path,
    batch_tag: Optional[str],
    spectrum_pooling: str,
    run_pooling: str,
    confidence_top_k: Optional[int],
    timestamp: str,
    completed_projects: List[Dict[str, Any]],
) -> None:
    """Write a manifest of all project outputs for one pooling pass."""
    tag = _sanitize_batch_tag(batch_tag) or "default"
    pooling_tag = pooling_artifact_suffix(spectrum_pooling, run_pooling, confidence_top_k)
    index_dir = output_root / tag / "_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{pooling_tag}_{timestamp}.json"
    payload = {
        "batch_tag": tag,
        "pooling": pooling_tag,
        "embedding_pooling": spectrum_pooling,
        "run_pooling": run_pooling,
        "confidence_top_k": confidence_top_k,
        "timestamp": timestamp,
        "n_projects": len(completed_projects),
        "projects": completed_projects,
    }
    with open(index_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote pooling run index → %s", index_path)


def _is_per_project_embeddings_cache(cache_root: Path) -> bool:
    """True when *cache_root* holds ``PXD*/embeddings.h5`` subdirs (not one combined h5)."""
    if (cache_root / "embeddings.h5").exists():
        return False
    return any(
        (child / "embeddings.h5").exists()
        for child in cache_root.iterdir()
        if child.is_dir() and child.name.startswith("PXD")
    )


def _group_precomputed_indices(
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, List[int]]]:
    projects_list, run_stems_list = _parse_project_run_from_metadata(metadata)
    grouped: Dict[str, Dict[str, List[int]]] = {}
    for idx in range(len(embeddings)):
        proj = projects_list[idx]
        stem = run_stems_list[idx]
        grouped.setdefault(proj, {}).setdefault(stem, []).append(idx)
    return grouped


def _validate_spectrum_cache_alignment(
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
    *,
    require_confidence: bool,
    context: str,
) -> None:
    """Ensure cached spectrum rows remain perfectly aligned across arrays."""
    if embeddings.ndim != 2:
        raise ValueError(f"[{context}] embeddings must be 2D (N, D), got {embeddings.shape}.")
    n_rows = embeddings.shape[0]
    if n_rows == 0:
        raise ValueError(f"[{context}] embeddings is empty.")
    if not metadata:
        raise ValueError(f"[{context}] metadata is empty.")

    for key, arr in metadata.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 0:
            raise ValueError(f"[{context}] metadata['{key}'] is scalar, expected row-aligned array.")
        if arr_np.shape[0] != n_rows:
            raise ValueError(
                f"[{context}] metadata['{key}'] row count {arr_np.shape[0]} != embeddings rows {n_rows}."
            )

    if "source_file" not in metadata and "usi" not in metadata:
        raise ValueError(f"[{context}] metadata must include 'source_file' or 'usi'.")

    parsed_projects, parsed_runs = _parse_project_run_from_metadata(metadata)
    if len(parsed_projects) != n_rows or len(parsed_runs) != n_rows:
        raise ValueError(
            f"[{context}] parsed run identity length mismatch "
            f"({len(parsed_projects)=}, {len(parsed_runs)=}, {n_rows=})."
        )

    if require_confidence:
        if "spectrum_confidence" not in metadata:
            raise ValueError(f"[{context}] missing required metadata['spectrum_confidence'].")
        conf = np.asarray(metadata["spectrum_confidence"]).reshape(-1)
        if conf.shape[0] != n_rows:
            raise ValueError(
                f"[{context}] spectrum_confidence rows {conf.shape[0]} != embeddings rows {n_rows}."
            )
        if not np.isfinite(conf).all():
            raise ValueError(f"[{context}] spectrum_confidence contains NaN/Inf.")


def _discover_projects_for_precomputed_cache(
    cache_root: Path,
    dataset_root: Path,
    project_filter: Optional[str],
) -> List[str]:
    if project_filter:
        return [project_filter]
    if _is_per_project_embeddings_cache(cache_root):
        return sorted(
            p.name
            for p in cache_root.iterdir()
            if p.is_dir() and p.name.startswith("PXD") and (p / "embeddings.h5").exists()
        )
    all_runs = discover_runs(dataset_root)
    return sorted({r.run_key.project for r in all_runs})


def _build_run_to_emb_from_precomputed_spectra(
    project: str,
    project_runs: Dict[str, List[int]],
    run_stems_to_process: List[str],
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
    args: argparse.Namespace,
    dataset_root: Path,
) -> Dict[str, Dict[str, Any]]:
    need_confidence = _needs_spectrum_confidence(args.run_pooling, args.confidence_top_k)
    spectrum_confidence = metadata.get("spectrum_confidence")
    if need_confidence and spectrum_confidence is None:
        raise ValueError(
            f"Run pooling '{args.run_pooling}' requires confidence, but no "
            "'spectrum_confidence' was found in the precomputed metadata."
        )

    run_to_emb: Dict[str, Dict[str, Any]] = {}
    for run_stem in run_stems_to_process:
        indices = np.array(project_runs[run_stem])
        run_embeddings = embeddings[indices]
        run_conf = spectrum_confidence[indices] if spectrum_confidence is not None else None

        pooled_embedding = pool_spectra_to_run_embedding(
            run_embeddings,
            run_pooling=args.run_pooling,
            spectrum_confidence=run_conf,
            confidence_top_k=args.confidence_top_k,
        )

        run_key = RunKey(condition=run_stem, project=project)
        key = run_key.as_str()

        r_conditions = {"condition_from_filename": run_stem}
        for pattern, label in _FILENAME_FRAG_PATTERNS:
            if pattern.search(run_stem):
                r_conditions["fragmentation_hint"] = label
                break
        frac = _FRAC_PATTERN.search(run_stem)
        if frac:
            r_conditions["fraction"] = int(frac.group(1))

        ipc_path = Path(dataset_root) / project / f"{run_stem}.ipc"
        if not ipc_path.exists():
            ipc_path = Path(dataset_root) / project / f"{run_stem}.mzML.ipc"

        if ipc_path.exists():
            try:
                ipc_conds = summarize_ipc_conditions(ipc_path)
                r_conditions.update(ipc_conds)
                r_conditions["ipc_path"] = str(ipc_path)
            except Exception as e:
                logger.debug(f"Could not read IPC conditions for {run_stem}: {e}")

        run_to_emb[key] = {
            "run_key": list(run_key.as_tuple()),
            "condition": run_key.condition,
            "project": run_key.project,
            "embedding": pooled_embedding,
            "n_spectra_embedded": len(indices),
            "n_spectra_pooled": min(args.confidence_top_k or len(indices), len(indices))
            if args.run_pooling == "topk_concat" or args.confidence_top_k is not None
            else len(indices),
            "embedding_dim": int(pooled_embedding.shape[0]),
            "conditions": r_conditions,
        }
    return run_to_emb


def _write_spectrum_cache_manifest(cache_root: Path, projects: List[str], args: argparse.Namespace) -> None:
    payload = {
        "task": "spectrum-level embedding cache for runs classification",
        "embedding_pooling": args.embedding_pooling,
        "checkpoint_path": args.checkpoint_path,
        "dataset_root": args.dataset_root,
        "batch_tag": args.batch_tag,
        "projects": projects,
        "created_at": datetime.now().isoformat(),
    }
    with open(cache_root / "cache_manifest.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_run_to_embedding_map(
    run_to_emb: Dict[str, Dict[str, Any]],
    output_dir: Path,
    spectrum_pooling: str,
    run_pooling: str = "mean",
    confidence_top_k: Optional[int] = None,
) -> Dict[str, str]:
    """Persist run → embedding index as JSON (no tensor files in output dir)."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "run_to_embedding.json"

    keys = list(run_to_emb.keys())
    index = {
        "pooling": {
            "spectrum": spectrum_pooling,
            "run": run_pooling,
            "confidence_top_k": confidence_top_k,
        },
        "runs": {
            k: {
                "run_key": run_to_emb[k]["run_key"],
                "condition": run_to_emb[k]["condition"],
                "project": run_to_emb[k]["project"],
                "n_spectra_embedded": run_to_emb[k]["n_spectra_embedded"],
                "n_spectra_pooled": run_to_emb[k]["n_spectra_pooled"],
                "embedding_dim": run_to_emb[k]["embedding_dim"],
            }
            for k in keys
        },
    }
    with open(json_path, "w") as f:
        json.dump(index, f, indent=2)

    return {"json": str(json_path)}


def attach_run_conditions(
    runs: List[RunRecord],
    run_to_emb: Dict[str, Dict[str, Any]],
) -> None:
    for run in runs:
        key = run.run_key.as_str()
        if key in run_to_emb:
            run_to_emb[key]["conditions"] = run.conditions


def _load_manifest_condition_lookup(
    dataset_root: Path,
) -> tuple[Dict[tuple[str, str], str], Dict[str, str]]:
    """Load ``(project, run_stem) → condition`` and ``project → condition_axis`` from manifest."""
    manifest_path = dataset_root / "manifest.yaml"
    lookup: Dict[tuple[str, str], str] = {}
    axes: Dict[str, str] = {}
    if not manifest_path.exists():
        logger.warning(
            "No manifest at %s — condition colouring falls back to IPC metadata",
            manifest_path,
        )
        return lookup, axes

    cfg = OmegaConf.load(manifest_path)
    for project_id, project in cfg.get("projects", {}).items():
        project_name = str(project_id)
        axes[project_name] = str(project.get("project_condition_axis", "condition"))
        for run_stem, run_info in project.get("runs", {}).items():
            cond = run_info.get("condition")
            if cond is not None:
                # Direct run key match (e.g. SCr1)
                lookup[(project_name, str(run_stem))] = str(cond)

                # Check pride_raw filename stem (e.g. 20240404_Jenny_Rucha_SCr1)
                pride_raw = run_info.get("pride_raw")
                if pride_raw:
                    raw_stem = Path(pride_raw).stem
                    lookup[(project_name, raw_stem)] = str(cond)

                # Check local_file filename stem
                local_file = run_info.get("local_file")
                if local_file:
                    lf_stem = Path(local_file).name
                    if lf_stem.endswith(".ipc"):
                        lf_stem = lf_stem[:-4]
                    if lf_stem.endswith(".mzML"):
                        lf_stem = lf_stem[:-5]
                    lookup[(project_name, lf_stem)] = str(cond)
    return lookup, axes


def _experimental_condition_label(
    run_entry: Dict[str, Any],
    manifest_lookup: Dict[tuple[str, str], str],
) -> str:
    """Best available experimental condition label for colouring within a project."""
    project = run_entry["project"]
    stem = run_entry["condition"]
    manifest_label = manifest_lookup.get((project, stem))
    if manifest_label and manifest_label not in ("TBD", "null", "None"):
        return manifest_label

    conds = run_entry.get("conditions", {})
    for key in ("fragmentation_hint", "dominant_frag_type"):
        if key in conds and conds[key]:
            return str(conds[key])
    return stem


def _short_run_label(condition_stem: str) -> str:
    """Short point label for plots (e.g. ``frac10`` or ``477-1``)."""
    frac = _FRAC_PATTERN.search(condition_stem)
    if frac:
        return f"frac{frac.group(1)}"
    return condition_stem


def _upload_to_s3(output_dir: Path, enabled: bool = False) -> None:
    """Upload all files in output_dir to S3 if enabled."""
    if not enabled:
        return
    s3 = S3FileHandler()
    if s3.s3 is None:
        logger.warning("S3 upload requested but S3FileHandler could not be initialized (no AWS credentials). Skipping upload.")
        return
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    logger.info(f"Uploading {len(files)} files from {output_dir} to S3...")
    for f in files:
        if S3FileHandler._aichor_enabled():
            s3_path = S3FileHandler.convert_to_s3_output(str(f))
        else:
            # Fallback for interactive sessions: upload to the inputs/runs bucket in the requested folder
            rel_path = f.relative_to(output_dir.parent)
            s3_path = f"s3://<inputs-bucket>/runs/{rel_path}"
        s3.upload(str(f), s3_path)
        logger.info(f"Uploaded {f.name} → {s3_path}")
    logger.info("S3 upload complete")


def upload_directory_to_s3(output_dir: Path, *, enabled: bool = True) -> None:
    """Public wrapper for uploading an entire results directory tree to S3."""
    _upload_to_s3(output_dir, enabled=enabled)


def _parse_project_run_from_metadata(metadata: Dict[str, np.ndarray]) -> tuple[List[str], List[str]]:
    projects = []
    run_stems = []

    n_samples = len(next(iter(metadata.values())))

    # Try source_file first
    if "source_file" in metadata:
        source_files = metadata["source_file"]
        for sf in source_files:
            if isinstance(sf, bytes):
                sf = sf.decode("utf-8")
            p = Path(sf)
            projects.append(p.parent.name)
            run_stems.append(_ipc_stem(p))
        return projects, run_stems

    # Try usi next
    if "usi" in metadata:
        usis = metadata["usi"]
        for usi in usis:
            if isinstance(usi, bytes):
                usi = usi.decode("utf-8")
            parts = usi.split(":")
            if len(parts) >= 3:
                projects.append(parts[1])
                run_stems.append(parts[2])
            else:
                projects.append("unknown")
                run_stems.append("unknown")
        return projects, run_stems

    raise ValueError(f"Metadata must contain either 'source_file' or 'usi' to identify project and run keys. Available keys: {list(metadata.keys())}")


def _fit_pca(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = min(2, embeddings.shape[0])
    return PCA(n_components=n_components).fit_transform(embeddings).astype(np.float32)


def _try_fit_umap(
    embeddings: np.ndarray,
    n_runs: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
    project: str,
) -> Optional[np.ndarray]:
    if n_runs < 2:
        return None
    try:
        return _fit_umap(
            embeddings,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=seed,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("UMAP failed for %s (%s); PCA plot still saved", project, exc)
        return None


def _save_projection_coords(
    output_path: Path,
    keys: List[str],
    coords: np.ndarray,
    prefix: str,
    color_labels: Optional[List[str]] = None,
    annotate_labels: Optional[List[str]] = None,
) -> None:
    with open(output_path, "w") as f:
        header = f"run_key,{prefix}_1,{prefix}_2"
        if color_labels is not None:
            header += ",condition_label"
        if annotate_labels is not None:
            header += ",run_label"
        f.write(header + "\n")
        for i, (rk, (x, y)) in enumerate(zip(keys, coords, strict=False)):
            row = f"{rk},{x},{y}"
            if color_labels is not None:
                row += f",{color_labels[i]}"
            if annotate_labels is not None:
                row += f",{annotate_labels[i]}"
            f.write(row + "\n")


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-12, None)
    return (normed @ normed.T).astype(np.float32)


def _fit_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    import umap

    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, 2), dtype=np.float32)

    n_neighbors = max(2, min(n_neighbors, n_samples - 2))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def _plot_run_embedding_space(
    embeddings: np.ndarray,
    annotate_labels: List[str],
    color_labels: List[str],
    title: str,
    output_path: Path,
    coords: np.ndarray,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
) -> None:
    unique = sorted(set(color_labels))
    cmap = plt.get_cmap("tab10", max(len(unique), 1))
    color_map = {lab: cmap(i) for i, lab in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (ann, lab) in enumerate(zip(annotate_labels, color_labels, strict=False)):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            c=[color_map.get(lab, "gray")],
            s=200,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            ann,
            (coords[i, 0], coords[i, 1]),
            fontsize=8,
            ha="center",
            xytext=(0, 8),
            textcoords="offset points",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(unique) <= 10:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[u],
                markersize=10,
                label=u,
            )
            for u in unique
        ]
        ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cluster_runs_in_embedding_space(
    run_to_emb: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dataset_root: Path,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
    artifact_suffix: str = "",
) -> Dict[str, Any]:
    """Within-project visualisation: PCA/UMAP figures and coordinate CSVs per project."""
    manifest_lookup, condition_axes = _load_manifest_condition_lookup(dataset_root)

    figures_dir = output_dir / "figures"
    coords_dir = output_dir / "coords"
    data_dir = output_dir / "data"
    for sub in (figures_dir, coords_dir, data_dir):
        sub.mkdir(parents=True, exist_ok=True)

    by_project: Dict[str, List[str]] = {}
    for key in run_to_emb:
        by_project.setdefault(run_to_emb[key]["project"], []).append(key)

    project_reports: Dict[str, Any] = {}
    plot_paths: List[str] = []

    for project in sorted(by_project):
        keys = sorted(by_project[project])
        embedding_dims = {run_to_emb[k]["embedding_dim"] for k in keys}
        if len(embedding_dims) > 1:
            logger.warning(
                "Project %s: variable embedding dimensions %s — skipping plot",
                project,
                sorted(embedding_dims),
            )
            continue

        embeddings = np.stack([run_to_emb[k]["embedding"] for k in keys], axis=0)
        n_runs = len(keys)
        if n_runs < 2:
            logger.info("Project %s: only %d run — skipping within-project plot", project, n_runs)
            continue
        condition_axis = condition_axes.get(project, "condition")
        color_labels = [_experimental_condition_label(run_to_emb[k], manifest_lookup) for k in keys]
        annotate_labels = [_short_run_label(run_to_emb[k]["condition"]) for k in keys]

        sim = _cosine_similarity_matrix(embeddings)
        cosine_path = data_dir / "run_cosine_similarity.json"
        with open(cosine_path, "w") as f:
            json.dump(
                {
                    "project": project,
                    "condition_axis": condition_axis,
                    "pooling_suffix": artifact_suffix,
                    "run_keys": keys,
                    "color_labels": color_labels,
                    "similarity": sim.tolist(),
                },
                f,
                indent=2,
            )

        pca_coords = _fit_pca(embeddings)
        umap_coords = _try_fit_umap(
            embeddings,
            n_runs,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
            seed,
            project,
        )

        pca_plot_path = figures_dir / "pca_by_condition.png"
        _plot_run_embedding_space(
            embeddings,
            annotate_labels,
            color_labels,
            f"{project} — runs coloured by {condition_axis} (within-project PCA)",
            pca_plot_path,
            coords=pca_coords,
            xlabel="PC1",
            ylabel="PC2",
        )
        pca_coords_path = coords_dir / "pca_coords.csv"
        _save_projection_coords(
            pca_coords_path,
            keys,
            pca_coords,
            "pc",
            color_labels=color_labels,
            annotate_labels=annotate_labels,
        )
        plot_paths.append(str(pca_plot_path))

        umap_plot_path: Optional[str] = None
        umap_coords_path: Optional[Path] = None
        if umap_coords is not None:
            umap_plot_path = str(figures_dir / "umap_by_condition.png")
            _plot_run_embedding_space(
                embeddings,
                annotate_labels,
                color_labels,
                f"{project} — runs coloured by {condition_axis} (within-project UMAP)",
                Path(umap_plot_path),
                coords=umap_coords,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
            )
            umap_coords_path = coords_dir / "umap_coords.csv"
            _save_projection_coords(
                umap_coords_path,
                keys,
                umap_coords,
                "umap",
                color_labels=color_labels,
                annotate_labels=annotate_labels,
            )
            plot_paths.append(umap_plot_path)

        logger.info(
            "Project %s: %d run(s), %d unique condition(s) → figures/%s%s",
            project,
            n_runs,
            len(set(color_labels)),
            pca_plot_path.name,
            f", figures/{Path(umap_plot_path).name}" if umap_plot_path else "",
        )

        project_reports[project] = {
            "n_runs": n_runs,
            "run_keys": keys,
            "condition_axis": condition_axis,
            "color_labels": color_labels,
            "n_unique_conditions": len(set(color_labels)),
            "pca_plot": str(pca_plot_path),
            "umap_plot": umap_plot_path,
            "pca_coords_csv": str(pca_coords_path),
            "umap_coords_csv": str(umap_coords_path) if umap_coords_path else None,
            "cosine_similarity_json": str(cosine_path),
            "pairwise_cosine_similarity": {f"{keys[i]} vs {keys[j]}": float(sim[i, j]) for i in range(n_runs) for j in range(i + 1, n_runs)},
        }

    return {
        "task": "within-project run clustering (PCA + UMAP plot per project)",
        "n_runs": len(run_to_emb),
        "n_projects": len(by_project),
        "projects": project_reports,
        "plots": plot_paths,
    }


def run_evaluation(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root)

    # Path A: Precomputed Embeddings
    if args.precomputed_embeddings_dir:
        precomputed_dir = args.precomputed_embeddings_dir
        if precomputed_dir.startswith("s3://"):
            logger.info(f"Downloading precomputed embeddings from S3: {precomputed_dir}")
            local_dir = Path(args.output_dir) / "precomputed_cache"
            local_dir.mkdir(parents=True, exist_ok=True)

            s3 = S3FileHandler()
            s3.download(f"{precomputed_dir.rstrip('/')}/embeddings.h5", str(local_dir / "embeddings.h5"))
            s3.download(f"{precomputed_dir.rstrip('/')}/index.faiss", str(local_dir / "index.faiss"))
            precomputed_dir = str(local_dir)

        cache_root = Path(precomputed_dir)
        per_project_cache = _is_per_project_embeddings_cache(cache_root)
        grouped_indices: Optional[Dict[str, Dict[str, List[int]]]] = None
        shared_embeddings: Optional[np.ndarray] = None
        shared_metadata: Optional[Dict[str, np.ndarray]] = None

        if per_project_cache:
            logger.info(f"Using per-project spectrum embedding cache under {cache_root}")
            projects = _discover_projects_for_precomputed_cache(cache_root, dataset_root, args.project)
        else:
            logger.info(f"Loading combined precomputed spectrum-level embeddings from {cache_root}")
            shared_embeddings, shared_metadata, _ = embedding_io.load(str(cache_root))
            _validate_spectrum_cache_alignment(
                shared_embeddings,
                shared_metadata,
                require_confidence=_needs_spectrum_confidence(args.run_pooling, args.confidence_top_k),
                context=str(cache_root),
            )
            grouped_indices = _group_precomputed_indices(shared_embeddings, shared_metadata)
            if args.project:
                projects = [args.project]
            else:
                projects = sorted(grouped_indices.keys())
            if not projects:
                raise FileNotFoundError("No projects found in precomputed embeddings metadata.")

        logger.info(f"Identified {len(projects)} project(s) to process: {projects}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_output_dir = Path(args.output_dir)
        completed_projects: List[Dict[str, Any]] = []

        for project in projects:
            if per_project_cache:
                proj_cache = cache_root / project
                if not (proj_cache / "embeddings.h5").exists():
                    logger.warning(f"No cached embeddings for project {project} at {proj_cache}, skipping.")
                    continue
                embeddings, metadata, _ = embedding_io.load(str(proj_cache))
                _validate_spectrum_cache_alignment(
                    embeddings,
                    metadata,
                    require_confidence=_needs_spectrum_confidence(args.run_pooling, args.confidence_top_k),
                    context=str(proj_cache),
                )
                project_runs = _group_precomputed_indices(embeddings, metadata).get(project, {})
                if not project_runs:
                    logger.warning(f"No runs found in cached embeddings for project {project}, skipping.")
                    continue
            else:
                assert grouped_indices is not None and shared_embeddings is not None and shared_metadata is not None
                if project not in grouped_indices:
                    logger.warning(f"Project {project} not found in precomputed embeddings, skipping.")
                    continue
                embeddings = shared_embeddings
                metadata = shared_metadata
                project_runs = grouped_indices[project]

            project_output_dir = resolve_project_output_dir(
                Path(args.output_dir),
                project,
                args.embedding_pooling,
                args.run_pooling,
                args.confidence_top_k,
                timestamp=timestamp,
                batch_tag=args.batch_tag,
            )
            last_output_dir = project_output_dir

            # Setup file logging
            log_file_path = project_output_dir / "runs_classification.log"
            file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)

            try:
                run_to_emb: Dict[str, Dict[str, Any]] = {}
                logger.info("\n==========================================")
                logger.info(f"Evaluating project (precomputed): {project}")
                logger.info("==========================================\n")

                run_stems_to_process = sorted(list(project_runs.keys()))

                if args.run_keys:
                    selected = {k.strip() for k in args.run_keys.split(",")}
                    run_stems_to_process = [r for r in run_stems_to_process if (r in selected or f"{r}::{project}" in selected)]
                if args.max_runs is not None:
                    run_stems_to_process = run_stems_to_process[: args.max_runs]

                if not run_stems_to_process:
                    logger.warning(f"No runs remaining for project {project} after filtering, skipping.")
                    continue

                logger.info(f"Selected {len(run_stems_to_process)} run(s) from project {project}")

                run_to_emb = _build_run_to_emb_from_precomputed_spectra(
                    project,
                    project_runs,
                    run_stems_to_process,
                    embeddings,
                    metadata,
                    args,
                    dataset_root,
                )

                artifact_suffix = pooling_artifact_suffix(
                    args.embedding_pooling,
                    args.run_pooling,
                    args.confidence_top_k,
                )
                embedding_artifacts = save_run_to_embedding_map(
                    run_to_emb,
                    project_output_dir,
                    spectrum_pooling=args.embedding_pooling,
                    run_pooling=args.run_pooling,
                    confidence_top_k=args.confidence_top_k,
                )

                with open(project_output_dir / "data" / "run_conditions.json", "w") as f:
                    json.dump(
                        {k: {"run_key": v["run_key"], **v.get("conditions", {})} for k, v in run_to_emb.items()},
                        f,
                        indent=2,
                    )

                cluster_report = cluster_runs_in_embedding_space(
                    run_to_emb,
                    project_output_dir,
                    dataset_root,
                    args.umap_n_neighbors,
                    args.umap_min_dist,
                    args.umap_metric,
                    args.seed,
                    artifact_suffix=artifact_suffix,
                )

                summary = {
                    "task": "cluster runs in embedding space (one embedding per .ipc run file)",
                    "n_runs": len(run_to_emb),
                    "run_keys": list(run_to_emb.keys()),
                    "checkpoint_path": args.checkpoint_path,
                    "dataset_root": str(dataset_root),
                    "precomputed_embeddings_dir": args.precomputed_embeddings_dir,
                    "embedding_pooling": args.embedding_pooling,
                    "run_pooling": args.run_pooling,
                    "confidence_top_k": args.confidence_top_k,
                    "confidence_temperature": args.confidence_temperature,
                    "batch_tag": args.batch_tag,
                    "max_spectra_per_run": args.max_spectra_per_run,
                    "cluster_report": cluster_report,
                    "command": "python " + " ".join(sys.argv),
                    "outputs": {
                        **embedding_artifacts,
                        "figures_pca": str(project_output_dir / "figures" / "pca_by_condition.png"),
                        "figures_umap": str(project_output_dir / "figures" / "umap_by_condition.png"),
                        "coords_pca": str(project_output_dir / "coords" / "pca_coords.csv"),
                        "coords_umap": str(project_output_dir / "coords" / "umap_coords.csv"),
                        "run_conditions": str(project_output_dir / "data" / "run_conditions.json"),
                    },
                }
                with open(project_output_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                completed_projects.append(
                    {
                        "project": project,
                        "output_dir": str(project_output_dir),
                        "n_runs": len(run_to_emb),
                        "figures": {
                            "pca": str(project_output_dir / "figures" / "pca_by_condition.png"),
                            "umap": str(project_output_dir / "figures" / "umap_by_condition.png"),
                        },
                        "coords": {
                            "pca": str(project_output_dir / "coords" / "pca_coords.csv"),
                            "umap": str(project_output_dir / "coords" / "umap_coords.csv"),
                        },
                    }
                )

            except Exception:
                logger.exception("An error occurred during evaluation of project %s", project)
                raise
            finally:
                root_logger.removeHandler(file_handler)
                file_handler.close()
                _upload_to_s3(project_output_dir, enabled=args.upload_to_s3)
                logger.info(f"Done project {project} — {len(run_to_emb)} run embedding(s) → {project_output_dir}")

        _write_pooling_run_index(
            Path(args.output_dir),
            args.batch_tag,
            args.embedding_pooling,
            args.run_pooling,
            args.confidence_top_k,
            timestamp,
            completed_projects,
        )
        if args.upload_to_s3 and completed_projects:
            _upload_to_s3(
                Path(args.output_dir) / (_sanitize_batch_tag(args.batch_tag) or "default") / "_indices",
                enabled=True,
            )
        return last_output_dir

    # Path B: Live inference
    if args.project:
        projects = [args.project]
    else:
        all_runs = discover_runs(dataset_root)
        projects = sorted(list({r.run_key.project for r in all_runs}))
        if not projects:
            raise FileNotFoundError(f"No IPC runs/projects found under {dataset_root}/PXD*/*.ipc")

    logger.info(f"Identified {len(projects)} project(s) to process: {projects}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint_cache = Path(args.output_dir) / "checkpoint_cache"

    model, processor, _ = _load_model_and_processor(
        args.checkpoint_path,
        Path(args.residues_config),
        device,
        checkpoint_cache,
    )

    if args.precompute_spectrum_embeddings_only:
        if not args.spectrum_embeddings_cache_dir:
            raise SystemExit(
                "error: --precompute-spectrum-embeddings-only requires --spectrum-embeddings-cache-dir"
            )
        cache_root = Path(args.spectrum_embeddings_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cached_projects: List[str] = []

        for project in projects:
            runs = discover_runs(dataset_root, project=project)
            if not runs:
                logger.warning(f"No runs found for project {project}, skipping.")
                continue
            if args.run_keys:
                selected = {k.strip() for k in args.run_keys.split(",")}
                runs = [
                    r
                    for r in runs
                    if (
                        r.run_key.as_str() in selected
                        or r.run_key.condition in selected
                        or f"{r.run_key.condition}::{r.run_key.project}" in selected
                    )
                ]
            if args.max_runs is not None:
                runs = runs[: args.max_runs]
            if not runs:
                logger.warning(f"No selected runs remaining for project {project}, skipping.")
                continue

            project_cache = cache_root / project
            logger.info(f"Precomputing spectrum embeddings for {project} ({len(runs)} runs) → {project_cache}")
            build_run_to_embedding_map(
                model,
                processor,
                runs,
                device,
                args.batch_size,
                args.max_spectra_per_run,
                args.embedding_pooling,
                args.run_pooling,
                args.confidence_top_k,
                args.confidence_temperature,
                args.seed,
                save_spectrum_embeddings_dir=project_cache,
            )
            cached_projects.append(project)
            logger.info(f"Cached spectrum embeddings for {project}")

        if not cached_projects:
            raise FileNotFoundError("No spectrum embeddings were cached for any project.")

        _write_spectrum_cache_manifest(cache_root, cached_projects, args)
        if args.upload_to_s3:
            _upload_to_s3(cache_root, enabled=True)
        logger.info(
            "Spectrum embedding precompute complete — %d project(s) at %s",
            len(cached_projects),
            cache_root,
        )
        return cache_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_output_dir = Path(args.output_dir)
    completed_projects: List[Dict[str, Any]] = []

    for project in projects:
        logger.info("\n==========================================")
        logger.info(f"Evaluating project: {project}")
        logger.info("==========================================\n")

        runs = discover_runs(dataset_root, project=project)
        if not runs:
            logger.warning(f"No runs found for project {project}, skipping.")
            continue

        if args.run_keys:
            selected = {k.strip() for k in args.run_keys.split(",")}
            runs = [
                r
                for r in runs
                if (r.run_key.as_str() in selected or r.run_key.condition in selected or f"{r.run_key.condition}::{r.run_key.project}" in selected)
            ]
        if args.max_runs is not None:
            runs = runs[: args.max_runs]

        if not runs:
            logger.warning(f"No selected runs remaining for project {project} after filtering, skipping.")
            continue

        project_output_dir = resolve_project_output_dir(
            Path(args.output_dir),
            project,
            args.embedding_pooling,
            args.run_pooling,
            args.confidence_top_k,
            timestamp=timestamp,
            batch_tag=args.batch_tag,
        )
        last_output_dir = project_output_dir

        # Setup file logging
        log_file_path = project_output_dir / "runs_classification.log"
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        try:
            logger.info(f"Selected {len(runs)} run(s) from project {project}")
            for r in runs:
                logger.info(f"  {r.run_key.as_tuple()}  ←  {r.source_file}")

            # Core step: run_key → embedding
            run_to_emb = build_run_to_embedding_map(
                model,
                processor,
                runs,
                device,
                args.batch_size,
                args.max_spectra_per_run,
                args.embedding_pooling,
                args.run_pooling,
                args.confidence_top_k,
                args.confidence_temperature,
                args.seed,
                save_spectrum_embeddings_dir=(
                    (Path(args.spectrum_embeddings_cache_dir) / project)
                    if args.spectrum_embeddings_cache_dir
                    else (project_output_dir / "data" if args.save_embeddings else None)
                ),
            )
            attach_run_conditions(runs, run_to_emb)
            artifact_suffix = pooling_artifact_suffix(
                args.embedding_pooling,
                args.run_pooling,
                args.confidence_top_k,
            )
            embedding_artifacts = save_run_to_embedding_map(
                run_to_emb,
                project_output_dir,
                spectrum_pooling=args.embedding_pooling,
                run_pooling=args.run_pooling,
                confidence_top_k=args.confidence_top_k,
            )

            # Write full conditions now that they are attached
            with open(project_output_dir / "data" / "run_conditions.json", "w") as f:
                json.dump(
                    {k: {"run_key": v["run_key"], **v.get("conditions", {})} for k, v in run_to_emb.items()},
                    f,
                    indent=2,
                )

            cluster_report = cluster_runs_in_embedding_space(
                run_to_emb,
                project_output_dir,
                dataset_root,
                args.umap_n_neighbors,
                args.umap_min_dist,
                args.umap_metric,
                args.seed,
                artifact_suffix=artifact_suffix,
            )

            summary = {
                "task": "cluster runs in embedding space (one embedding per .ipc run file)",
                "n_runs": len(runs),
                "run_keys": [r.run_key.as_str() for r in runs],
                "checkpoint_path": args.checkpoint_path,
                "dataset_root": str(dataset_root),
                "embedding_pooling": args.embedding_pooling,
                "run_pooling": args.run_pooling,
                "confidence_top_k": args.confidence_top_k,
                "confidence_temperature": args.confidence_temperature,
                "batch_tag": args.batch_tag,
                "max_spectra_per_run": args.max_spectra_per_run,
                "cluster_report": cluster_report,
                "command": "python " + " ".join(sys.argv),
                "outputs": {
                    **embedding_artifacts,
                    "figures_pca": str(project_output_dir / "figures" / "pca_by_condition.png"),
                    "figures_umap": str(project_output_dir / "figures" / "umap_by_condition.png"),
                    "coords_pca": str(project_output_dir / "coords" / "pca_coords.csv"),
                    "coords_umap": str(project_output_dir / "coords" / "umap_coords.csv"),
                    "run_conditions": str(project_output_dir / "data" / "run_conditions.json"),
                },
            }
            with open(project_output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            completed_projects.append(
                {
                    "project": project,
                    "output_dir": str(project_output_dir),
                    "n_runs": len(runs),
                    "figures": {
                        "pca": str(project_output_dir / "figures" / "pca_by_condition.png"),
                        "umap": str(project_output_dir / "figures" / "umap_by_condition.png"),
                    },
                    "coords": {
                        "pca": str(project_output_dir / "coords" / "pca_coords.csv"),
                        "umap": str(project_output_dir / "coords" / "umap_coords.csv"),
                    },
                }
            )

        except Exception:
            logger.exception("An error occurred during evaluation of project %s", project)
            raise
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()
            _upload_to_s3(project_output_dir, enabled=args.upload_to_s3)
            logger.info(f"Done project {project} — {len(runs)} run embedding(s) → {project_output_dir}")

    _write_pooling_run_index(
        Path(args.output_dir),
        args.batch_tag,
        args.embedding_pooling,
        args.run_pooling,
        args.confidence_top_k,
        timestamp,
        completed_projects,
    )
    if args.upload_to_s3 and completed_projects:
        _upload_to_s3(
            Path(args.output_dir) / (_sanitize_batch_tag(args.batch_tag) or "default") / "_indices",
            enabled=True,
        )
    return last_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Compute run_key=(condition, project) → embedding for each .ipc run, then cluster / visualise runs in embedding space."),
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="runs_classification_dataset root (PXD*/<run>.ipc)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project folder (starting with PXD) to target (optional)",
    )
    parser.add_argument(
        "--precomputed-embeddings-dir",
        default=None,
        help=(
            "Directory with pre-generated spectrum-level embeddings. "
            "Accepts one combined embeddings.h5 or per-project PXD*/embeddings.h5 layout."
        ),
    )
    parser.add_argument(
        "--spectrum-embeddings-cache-dir",
        default=None,
        help="Directory to write per-project spectrum embedding caches (PXD*/embeddings.h5).",
    )
    parser.add_argument(
        "--precompute-spectrum-embeddings-only",
        action="store_true",
        help=(
            "Only run FM inference and cache spectrum-level embeddings + confidence; "
            "skip run pooling, plots, and per-pooling result dirs."
        ),
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload evaluation results to S3 when running on AIchor (gated flag).",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save computed spectrum-level embeddings and FAISS index to the output directory (optional).",
    )
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default="instanovo/foundational/eval/run_condition_results",
    )
    parser.add_argument(
        "--batch-tag",
        default=None,
        help=(
            "Optional label embedded in output directory names (e.g. min-int-fix) so re-runs with the same pooling do not overwrite prior artifacts."
        ),
    )
    parser.add_argument(
        "--residues-config",
        default=str(Path(__file__).parents[2] / "configs" / "residues" / "default.yaml"),
    )
    parser.add_argument(
        "--embedding-pooling",
        choices=["cls", "mean_pool", "confidence"],
        default="mean_pool",
        help="Spectrum-level pooling inside the FM (per MS/MS scan)",
    )
    parser.add_argument(
        "--run-pooling",
        choices=list(RUN_POOLING_CHOICES),
        default="mean",
        help=(
            "Run-level pooling over spectrum embeddings: "
            "mean (768-d), weighted_mean (768-d, confidence-weighted), "
            "topk_concat (k×768-d, top-k by confidence then flatten)"
        ),
    )
    parser.add_argument(
        "--confidence-top-k",
        type=int,
        default=None,
        help=("Required for topk_concat. Optional for mean/weighted_mean: keep only the top-k spectra by classifier confidence before pooling"),
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for spectrum-level confidence pooling",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-spectra-per-run", type=int, default=None)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of runs (e.g. 2 for initial validation)",
    )
    parser.add_argument(
        "--run-keys",
        default=None,
        help="Comma-separated run keys or condition stems to select (optional)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    parser.add_argument("--umap-metric", default="cosine")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_pooling == "topk_concat" and args.confidence_top_k is None:
        raise SystemExit("error: --run-pooling topk_concat requires --confidence-top-k")
    if args.precompute_spectrum_embeddings_only and args.precomputed_embeddings_dir:
        raise SystemExit(
            "error: --precompute-spectrum-embeddings-only cannot be combined with --precomputed-embeddings-dir"
        )
    run_evaluation(args)


if __name__ == "__main__":
    main()
