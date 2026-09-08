"""Run Casanovo de novo peptide sequencing on foundation-model datasets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from instanovo.__init__ import console
from instanovo_fm.eval._extract_embeddings_common import (
    load_spectrum_dataframe,
    upload_to_s3,
)
from instanovo_fm.eval._predict_de_novo_common import (
    CASANOVO_TO_UNIMOD,
    MODEL_SCORING,
    _resolve_group,
    build_model_vocab,
    build_prediction_dataframe,
    filter_unpredictable_rows,
    log_supported_residues,
    reconcile_max_charge,
    save_predictions,
    score_predictions,
)
from instanovo_fm.utils.checkpoints import resolve_checkpoint
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


DEFAULT_MAX_CHARGE = 4


def _require_casanovo() -> Tuple[Any, Any, float]:
    """Import Casanovo's Spec2Pep, spectrum_utils.spectrum, and Casanovo's proton-mass constant.

    The proton mass derives neutral precursor mass from m/z (``(m/z - PROTON) * charge``).

    Returns:
        Tuple ``(Spec2Pep, spectrum_utils.spectrum, PROTON)``.

    Raises:
        ImportError: If casanovo (and its dependencies) are not installed.
    """
    try:
        import spectrum_utils.spectrum as sus  # type: ignore[import]
        from casanovo.data.db_utils import PROTON  # type: ignore[import]
        from casanovo.denovo.model import Spec2Pep  # type: ignore[import]

        return Spec2Pep, sus, PROTON
    except ImportError as exc:
        raise ImportError("casanovo is not installed. Install it with:\n    uv pip install casanovo\nthen re-run this script.") from exc


def _casanovo_preprocessing(max_charge: int, min_intensity: Optional[float]) -> Tuple[List[Any], Any]:
    """Source Casanovo's exact native preprocessing chain + valid-charge set from its own DeNovoDataModule.

    Args:
        max_charge: Maximum precursor charge, read from the checkpoint.
        min_intensity: Optional intensity-threshold override (None = Casanovo's default of 0.01).

    Returns:
        Tuple ``(preprocessing_fn, valid_charge)`` — the ordered ``spectrum -> spectrum`` chain and the
        array of accepted precursor charges.
    """
    from casanovo.denovo.dataloaders import DeNovoDataModule  # type: ignore[import]

    kwargs: Dict[str, Any] = {"lance_dir": "", "max_charge": max_charge}
    if min_intensity is not None:
        kwargs["min_intensity"] = min_intensity
    data_module = DeNovoDataModule(**kwargs)
    return data_module.preprocessing_fn, data_module.valid_charge


def _preprocess_spectrum(row: Any, sus: Any, preprocessing_fn: List[Any], valid_charge: Any) -> Optional[Any]:
    """Build an ``MsmsSpectrum`` for one row and apply Casanovo's native pipeline.

    Args:
        row: A pandas row with ``mz_array``, ``intensity_array``, ``precursor_mz``, ``precursor_charge``.
        sus: The ``spectrum_utils.spectrum`` module.
        preprocessing_fn: The ordered Casanovo preprocessing chain (from :func:`_casanovo_preprocessing`).
        valid_charge: Array of accepted precursor charges (spectra outside it are unpredictable).

    Returns:
        The preprocessed ``MsmsSpectrum``, or ``None`` if the spectrum is unpredictable (charge out of
        range, or discarded by the low-quality / intensity filters).
    """
    charge = int(row["precursor_charge"])
    if charge not in valid_charge:
        return None
    try:
        spectrum = sus.MsmsSpectrum(
            str(row.get("prediction_id", "")),
            float(row["precursor_mz"]),
            charge,
            np.asarray(row["mz_array"], dtype=np.float64),
            np.asarray(row["intensity_array"], dtype=np.float32),
        )
        for fn in preprocessing_fn:
            spectrum = fn(spectrum)
    except ValueError:
        return None
    return spectrum


def _decode_batch(
    model: Any,
    buffer: List[Tuple[int, Any, float, int]],
    torch_device: torch.device,
    records_by_id: Dict[int, Dict[str, Any]],
    proton_mass: float,
) -> None:
    """Beam-search-decode one batch of preprocessed spectra and fill their prediction records.

    Args:
        model: The loaded ``Spec2Pep`` in eval mode.
        buffer: List of ``(prediction_id, MsmsSpectrum, precursor_mz, charge)`` for the batch.
        torch_device: Device to run decoding on.
        records_by_id: Prediction-record dict (mutated in place) keyed by prediction_id.
        proton_mass: Casanovo's proton-mass constant, used to derive neutral precursor mass from m/z.
    """
    if not buffer:
        return
    batch_size = len(buffer)
    max_len = max(len(s.mz) for _, s, _, _ in buffer)
    mzs = torch.zeros(batch_size, max_len)
    intensities = torch.zeros(batch_size, max_len)
    precursors = torch.zeros(batch_size, 3)
    for j, (_pid, spectrum, prec_mz, charge) in enumerate(buffer):
        n = len(spectrum.mz)
        mzs[j, :n] = torch.from_numpy(np.ascontiguousarray(spectrum.mz, dtype=np.float32))
        intensities[j, :n] = torch.from_numpy(np.ascontiguousarray(spectrum.intensity, dtype=np.float32))
        precursors[j] = torch.tensor([(prec_mz - proton_mass) * charge, float(charge), prec_mz])

    mzs, intensities, precursors = mzs.to(torch_device), intensities.to(torch_device), precursors.to(torch_device)
    with torch.inference_mode():
        preds = model.beam_search_decode(mzs, intensities, precursors)

    for (pid, _s, _mz, _c), spectrum_preds in zip(buffer, preds, strict=False):
        if spectrum_preds:
            peptide_score, _aa_scores, peptide = spectrum_preds[0]
            # Written in Casanovo's own notation; remapped to UNIMOD at scoring via MODEL_SCORING.
            records_by_id[pid]["predictions"] = peptide
            records_by_id[pid]["log_probs"] = float(math.log(max(float(peptide_score), 1e-10)))


@dataclass(frozen=True)
class LoadedCasanovo:
    """A loaded Casanovo model plus everything :func:`predict_dataframe` needs.

    Load once (via :func:`load_casanovo_model`) and reuse across many datasets — the pipeline
    runner loops all validation datasets with a single loaded model.

    Attributes:
        model: The eval-mode ``Spec2Pep`` on ``torch_device``.
        sus: The ``spectrum_utils.spectrum`` module.
        proton_mass: Casanovo's proton-mass constant.
        preprocessing_fn: Casanovo's native preprocessing chain.
        valid_charge: Array of accepted precursor charges.
        torch_device: Device the model runs on.
        model_vocab: Canonical residues Casanovo can emit (targets outside it are dropped upfront).
        max_charge: Effective maximum precursor charge (config-reconciled against the checkpoint's).
    """

    model: Any
    sus: Any
    proton_mass: float
    preprocessing_fn: List[Any]
    valid_charge: Any
    torch_device: torch.device
    model_vocab: set[str]
    max_charge: int


def load_casanovo_model(
    checkpoint_path: str,
    device: str = "cuda",
    *,
    checkpoint_url: Optional[str] = None,
    n_beams: Optional[int] = None,
    min_intensity: Optional[float] = None,
    max_charge: Optional[int] = None,
) -> LoadedCasanovo:
    """Resolve the checkpoint, load the full ``Spec2Pep``, and source Casanovo's native preprocessing.

    Args:
        checkpoint_path: Local (or ``s3://``) path to the Casanovo checkpoint (.ckpt).
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        checkpoint_url: If provided and ``checkpoint_path`` doesn't exist, download from here.
        n_beams: Override the checkpoint's beam width (default: use the checkpoint's value).
        min_intensity: Override Casanovo's native intensity threshold (default: 0.01).
        max_charge: Optional config override, reconciled against the checkpoint's ``max_charge`` (the
            smaller wins; a larger config value is clamped down with a warning).

    Returns:
        A :class:`LoadedCasanovo` bundle ready for :func:`predict_dataframe`.
    """
    spec2pep_cls, sus, proton_mass = _require_casanovo()

    ckpt = resolve_checkpoint(
        checkpoint_path,
        checkpoint_url,
        not_found_hint="  https://github.com/Noble-Lab/casanovo/releases",
    )
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading Casanovo checkpoint from {ckpt}")
    model = spec2pep_cls.load_from_checkpoint(str(ckpt), map_location="cpu", weights_only=False)
    model = model.to(torch_device)
    model.eval()
    if n_beams is not None:
        model.n_beams = n_beams
    model_max_charge = int(model.hparams.get("max_charge", DEFAULT_MAX_CHARGE))
    effective_max_charge = reconcile_max_charge(max_charge, model_max_charge, "Casanovo")
    logger.info(f"Casanovo loaded (n_beams={model.n_beams}, max_peptide_len={model.max_peptide_len}, max_charge={effective_max_charge})")
    log_supported_residues(model.tokenizer.residues, "Casanovo")

    if min_intensity is not None:
        logger.info(f"Native preprocessing with min_intensity override = {min_intensity}")
    preprocessing_fn, valid_charge = _casanovo_preprocessing(effective_max_charge, min_intensity)

    model_vocab = build_model_vocab(list(model.tokenizer.residues.keys()), CASANOVO_TO_UNIMOD)

    return LoadedCasanovo(model, sus, proton_mass, preprocessing_fn, valid_charge, torch_device, model_vocab, effective_max_charge)


def predict_dataframe(df: Any, loaded: LoadedCasanovo, batch_size: int = 64) -> Any:
    """Decode a loaded spectra DataFrame into a prediction DataFrame (canonical schema).

    Args:
        df: A spectra DataFrame (from :func:`load_spectrum_dataframe`) with ``mz_array``,
            ``intensity_array``, ``precursor_mz``, ``precursor_charge`` and (optionally) ``sequence``.
        loaded: A :class:`LoadedCasanovo` bundle.
        batch_size: Predictable spectra per beam-search call.

    Returns:
        A prediction DataFrame in the canonical schema (see ``PREDICTION_CSV_COLUMNS``).
    """
    has_targets = "sequence" in df.columns
    if not has_targets:
        logger.warning("No 'sequence' column found — writing predictions with empty targets (de novo mode).")

    df = filter_unpredictable_rows(df, max_charge=loaded.max_charge, model_vocab=loaded.model_vocab, model_name="Casanovo")
    columns = list(df.columns)
    target_strings = [str(t) if t is not None else "" for t in df["sequence"]] if has_targets else [""] * len(df)

    records_by_id: Dict[int, Dict[str, Any]] = {}
    buffer: List[Tuple[int, Any, float, int]] = []
    n_predictable = 0
    n_spectrum_drop = 0

    for prediction_id, (_idx, row) in enumerate(df.iterrows()):  # TODO inefficient?
        charge = int(row["precursor_charge"])
        prec_mz = float(row["precursor_mz"])
        spectrum = _preprocess_spectrum(row, loaded.sus, loaded.preprocessing_fn, loaded.valid_charge)
        spectrum_ok = spectrum is not None
        records_by_id[prediction_id] = {
            "prediction_id": prediction_id,
            "predictions": "",
            "targets": target_strings[prediction_id],
            "log_probs": float("nan"),
            "predictable": spectrum_ok,
            "precursor_mz": prec_mz,
            "precursor_charge": charge,
            "group": _resolve_group(row, columns),
        }
        if spectrum_ok:
            n_predictable += 1
            buffer.append((prediction_id, spectrum, prec_mz, charge))
            if len(buffer) >= batch_size:
                _decode_batch(loaded.model, buffer, loaded.torch_device, records_by_id, loaded.proton_mass)
                buffer = []
        else:
            n_spectrum_drop += 1
    _decode_batch(loaded.model, buffer, loaded.torch_device, records_by_id, loaded.proton_mass)

    logger.info(
        f"Decoded {n_predictable:,}/{len(df):,} spectra "
        f"({n_spectrum_drop:,} discarded by preprocessing, counted as misses under the true denominator)"
    )
    return build_prediction_dataframe(list(records_by_id.values()))


def predict(
    parquet_path: str,
    output_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    checkpoint_url: Optional[str] = None,
    n_beams: Optional[int] = None,
    score: bool = False,
    min_intensity: Optional[float] = None,
    max_charge: Optional[int] = None,
) -> None:
    """Run Casanovo de novo sequencing on one dataset and write predictions in the canonical CSV schema.

    Args:
        parquet_path: Path to input parquet file(s) (or ``s3://`` URI).
        output_path: Path to write the prediction CSV.
        checkpoint_path: Local (or ``s3://``) path to the Casanovo checkpoint (.ckpt).
        batch_size: Inference batch size (predictable spectra per beam-search call).
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to load.
        checkpoint_url: If provided and ``checkpoint_path`` doesn't exist, download from here.
        n_beams: Override the checkpoint's beam width (default: use the checkpoint's value).
        score: If True, score the predictions in-process with PredictionAnalyzer after writing.
        min_intensity: Override Casanovo's native intensity threshold (default: 0.01).
        max_charge: Optional config override, reconciled against the checkpoint's ``max_charge``.
    """
    loaded = load_casanovo_model(
        checkpoint_path, device, checkpoint_url=checkpoint_url, n_beams=n_beams, min_intensity=min_intensity, max_charge=max_charge
    )
    df = load_spectrum_dataframe(parquet_path, max_samples)
    has_targets = "sequence" in df.columns
    pred_df = predict_dataframe(df, loaded, batch_size)
    save_predictions(pred_df, output_path)

    if score:
        if has_targets:
            score_predictions(output_path, **MODEL_SCORING["casanovo"])
        else:
            logger.warning("--score requested but data has no targets; skipping scoring.")

    from pathlib import Path

    upload_to_s3(Path(output_path).parent)


def main() -> None:
    """Run Casanovo de novo sequencing from the command line."""
    parser = argparse.ArgumentParser(description="Run Casanovo de novo sequencing into the canonical prediction-CSV schema")
    parser.add_argument("--parquet_path", required=True, help="Path to input parquet file or glob pattern (or s3:// URI)")
    parser.add_argument("--output_path", required=True, help="Path to write the prediction CSV")
    parser.add_argument("--checkpoint_path", required=True, help="Local or s3:// path to the Casanovo checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", default="cuda", help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on number of spectra to load (default: all)")
    parser.add_argument(
        "--checkpoint_url",
        default=None,
        help="URL to download the checkpoint if --checkpoint_path does not exist. Releases: https://github.com/Noble-Lab/casanovo/releases",
    )
    parser.add_argument("--n_beams", type=int, default=None, help="Override beam width (default: checkpoint value, usually 1)")
    parser.add_argument("--score", action="store_true", help="Score predictions with PredictionAnalyzer after writing")
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=None,
        help="Override Casanovo's native intensity threshold (default: 0.01)",
    )
    parser.add_argument(
        "--max_charge",
        type=int,
        default=None,
        help="Optional max precursor charge, reconciled against the checkpoint's (default: checkpoint value)",
    )
    args = parser.parse_args()

    predict(
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        checkpoint_url=args.checkpoint_url,
        n_beams=args.n_beams,
        score=args.score,
        min_intensity=args.min_intensity,
        max_charge=args.max_charge,
    )


if __name__ == "__main__":
    main()
