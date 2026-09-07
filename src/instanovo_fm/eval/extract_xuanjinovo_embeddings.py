"""Extract XuanjiNovo spectrum encoder embeddings into the foundation model HDF5 format.

Usage:
    uv run python -m instanovo_fm.eval.extract_xuanjinovo_embeddings \
        --parquet_path /path/to/data.parquet \
        --output_dir /path/to/output/xuanjinovo \
        --checkpoint_path /path/to/XuanjiNovo_130M_massnet_massivekb.ckpt \
        [--batch_size 64] \
        [--device cuda] \
        [--max_samples 10000] \
        [--checkpoint_url https://...]

Encoder isolation:
    Only XuanjiNovo's ``SpectrumEncoder`` is loaded (a vendored, self-contained copy — see
    ``_xuanjinovo_encoder.py``). ``encoder(spectra, precursors)`` returns memories of shape
    (B, n_peaks+1, D); the first token (memories[:, 0, :]) is the contextualised precursor token
    and the remaining tokens are the peak tokens.

m/z denormalisation:
    FoundationalDataProcessor normalises m/z to [0, 1] by dividing by max_mz=2500. This script
    multiplies spectra[:, :, 0] by max_mz before the encoder, which expects raw (Da) m/z values.

Preprocessing note:
    Like the Casanovo baseline, this uses the shared FoundationalDataProcessor pipeline (200
    peaks, no precursor-peak removal) rather than XuanjiNovo's native pipeline (800 peaks,
    min_mz=1/max_mz=6500, remove_precursor_tol=1). This keeps XuanjiNovo directly comparable to
    the other baselines; native-per-model preprocessing is a possible later follow-up.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import torch

from instanovo.__init__ import console
from instanovo_fm.eval._extract_embeddings_common import (
    FAISS_AVAILABLE,
    build_dataloader,
    collect_batch_metadata,
    finalize_and_save,
)
from instanovo_fm.eval._xuanjinovo_encoder import load_xuanjinovo_encoder
from instanovo_fm.utils.checkpoints import resolve_checkpoint
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# XuanjiNovo's charge_encoder is nn.Embedding(10, dim) indexed by (charge - 1), so charges must
# lie in [1, 10] to avoid out-of-range / negative embedding indices.
MAX_CHARGE = 10

CHECKPOINT_URL = "https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_130M_massnet_massivekb.ckpt"

# XuanjiNovo's native spectrum preprocessing (MassNet-DDA XuanjiNovo/config.yaml). Selected via
# --preprocessing native; m/z is kept in raw Da (no normalisation / denormalisation round-trip).
NATIVE_PREPROCESSING: dict[str, Any] = {
    "n_peaks": 800,
    "min_mz": 1.0,
    "max_mz": 6500.0,
    "min_intensity": 0.0,
    "remove_precursor_tol": 1.0,
}


def extract(
    parquet_path: str,
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    checkpoint_url: Optional[str] = None,
    num_workers: int = 4,
    pooling: str = "mean_peaks",
    search_data_path: str = "src/data/search_data.xlsx",
    max_mz: float = 2500.0,
    preprocessing: str = "foundation",
    min_intensity: Optional[float] = None,
) -> None:
    """Extract XuanjiNovo encoder embeddings and save in foundation-model HDF5 format.

    Args:
        parquet_path: Path to input parquet file(s).
        output_dir: Directory to write embeddings.h5 and index.faiss.
        checkpoint_path: Local path to the XuanjiNovo checkpoint (.ckpt).
        batch_size: Inference batch size.
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to embed.
        checkpoint_url: If provided and checkpoint_path doesn't exist, download from here.
        num_workers: DataLoader worker count.
        pooling: Pooling strategy — "mean_peaks" (mean over non-padding peak tokens, default) or
            "precursor" (the contextualised precursor token at position 0).
        search_data_path: Path to the search-data table for USI-based search metadata lookup.
        max_mz: m/z normalisation divisor for the foundation preprocessing (the foundation model's
            ``max_mz``). Ignored when ``preprocessing="native"`` (native m/z range is used instead).
        preprocessing: "foundation" (shared FoundationalDataProcessor, 200 peaks, normalised m/z,
            default) or "native" (XuanjiNovo's own preprocessing — see ``NATIVE_PREPROCESSING`` —
            with raw m/z).
        min_intensity: Optional override for the preprocessing intensity threshold. If None, the
            mode default is used (foundation: 0.01; native: ``NATIVE_PREPROCESSING['min_intensity']``).

    Raises:
        ValueError: If ``pooling`` or ``preprocessing`` is not a recognised value.
        ImportError: If faiss is not available.
        FileNotFoundError: If the checkpoint is missing and no ``checkpoint_url`` is given.
    """
    if pooling not in ("mean_peaks", "precursor"):
        raise ValueError(f"pooling must be 'mean_peaks' or 'precursor', got {pooling!r}")
    if preprocessing not in ("foundation", "native"):
        raise ValueError(f"preprocessing must be 'foundation' or 'native', got {preprocessing!r}")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    # ------------------------------------------------------------------ #
    # 1. Resolve checkpoint                                                #
    # ------------------------------------------------------------------ #
    ckpt = resolve_checkpoint(checkpoint_path, checkpoint_url, not_found_hint=f"  {CHECKPOINT_URL}")

    # ------------------------------------------------------------------ #
    # 2. Load model (encoder only)                                         #
    # ------------------------------------------------------------------ #
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading XuanjiNovo encoder from {ckpt}")
    encoder, _ = load_xuanjinovo_encoder(str(ckpt), torch_device)
    logger.info("XuanjiNovo encoder loaded")

    # ------------------------------------------------------------------ #
    # 3. Build DataLoader                                                  #
    # ------------------------------------------------------------------ #
    if preprocessing == "native":
        native_params = dict(NATIVE_PREPROCESSING)
        if min_intensity is not None:
            native_params["min_intensity"] = min_intensity
        logger.info(f"Using XuanjiNovo-native preprocessing (raw m/z): {native_params}")
        dataloader = build_dataloader(
            parquet_path,
            batch_size,
            max_samples,
            num_workers,
            normalize_mz=False,
            **native_params,
        )
    else:
        extra: dict[str, Any] = {} if min_intensity is None else {"min_intensity": min_intensity}
        if min_intensity is not None:
            logger.info(f"Foundation preprocessing with min_intensity override = {min_intensity}")
        dataloader = build_dataloader(parquet_path, batch_size, max_samples, num_workers, max_mz=max_mz, **extra)

    # ------------------------------------------------------------------ #
    # 4. Embedding loop (XuanjiNovo-specific encode + pool)                #
    # ------------------------------------------------------------------ #
    all_embeddings: list = []
    all_metadata: list = []
    total_processed = 0

    with torch.inference_mode():
        for batch in dataloader:
            spectra = batch["spectra"].to(torch_device)  # (B, n_peaks, 2)
            precursors = batch["precursors"].to(torch_device)  # (B, 3): [mass, charge, mz]

            # Denormalise m/z for the foundation pipeline (it divides by max_mz). The native
            # pipeline already supplies raw m/z, so no denormalisation is needed there.
            spectra_raw = spectra.clone()
            if preprocessing == "foundation":
                spectra_raw[:, :, 0] = spectra_raw[:, :, 0] * max_mz

            # Clamp charge to the encoder's embedding range to avoid index errors.
            precursors = precursors.clone()
            precursors[:, 1] = precursors[:, 1].clamp(min=1, max=MAX_CHARGE)  # TODO perhaps refactor for all extraction code, drop instead?

            # TODO precursor mixed in to encoder step so we can't just chop first token
            # Run XuanjiNovo encoder → (B, n_peaks+1, D), layout [precursor(0) | peak_1 ... peak_N].
            memories, mem_mask = encoder(spectra_raw, precursors)

            # Guard the assumed token layout (one prepended precursor token). Fails loudly if a
            # future XuanjiNovo encoder changes it.
            assert memories.shape[1] == spectra_raw.shape[1] + 1, (
                f"Unexpected XuanjiNovo encoder layout: {memories.shape[1]} tokens for {spectra_raw.shape[1]} peaks "
                "(expected n_peaks+1 — one prepended precursor token). XuanjiNovo encoder may have changed."
            )

            if pooling == "precursor":
                embedding = memories[:, 0, :]  # (B, D)
            else:
                # mean_peaks: average over non-padding peak tokens (positions 1+), using the
                # encoder's own padding mask (True = padding).
                peak_tokens = memories[:, 1:, :]  # (B, N_peaks, D)
                valid = (~mem_mask[:, 1:]).unsqueeze(-1).float()  # (B, N_peaks, 1)
                embedding = (peak_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            emb_np = embedding.cpu().float().numpy()
            batch_size_actual = len(emb_np)

            all_embeddings.append(emb_np)
            all_metadata.append(collect_batch_metadata(batch, total_processed, batch_size_actual))
            total_processed += batch_size_actual

    logger.info(f"Embedded {total_processed:,} spectra")

    # ------------------------------------------------------------------ #
    # 5. Post-hoc metadata + FAISS + save + S3 upload (shared)            #
    # ------------------------------------------------------------------ #
    finalize_and_save(
        all_embeddings,
        all_metadata,
        search_data_path=search_data_path,
        output_dir=output_dir,
        pooling=pooling,
    )


def main() -> None:
    """Extract XuanjiNovo embeddings."""
    parser = argparse.ArgumentParser(description="Extract XuanjiNovo encoder embeddings into foundation-model HDF5 format")
    parser.add_argument("--parquet_path", required=True, help="Path to input parquet file or glob pattern")
    parser.add_argument("--output_dir", required=True, help="Directory to write embeddings.h5 and index.faiss")
    parser.add_argument("--checkpoint_path", required=True, help="Local path to XuanjiNovo checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", default="cuda", help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on number of spectra to embed (default: all)")
    parser.add_argument(
        "--checkpoint_url",
        default=None,
        help=f"URL to download the checkpoint if --checkpoint_path does not exist. Default mirror: {CHECKPOINT_URL}",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument(
        "--pooling",
        default="mean_peaks",
        choices=["mean_peaks", "precursor"],
        help="Pooling strategy: 'mean_peaks' (mean over peak tokens, default) or 'precursor' (position-0 token)",
    )
    parser.add_argument(
        "--search_data_path",
        default="src/data/search_data.xlsx",
        help="Path to the search-data table for USI-based search metadata lookup",
    )
    parser.add_argument(
        "--max_mz",
        type=float,
        default=2500.0,
        help="m/z normalisation divisor (foundation model's max_mz). The baseline scripts source "
        "this from the model config; default 2500.0 for direct invocation. Ignored when --preprocessing native.",
    )
    parser.add_argument(
        "--preprocessing",
        default="foundation",
        choices=["foundation", "native"],
        help="Spectrum preprocessing: 'foundation' (shared FoundationalDataProcessor, default) or "
        "'native' (XuanjiNovo's own 800-peak / 1-6500 m/z / precursor-removal pipeline)",
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=None,
        help="Override the preprocessing intensity threshold (default: mode default — foundation 0.01, native 0.0)",
    )
    args = parser.parse_args()

    extract(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        checkpoint_url=args.checkpoint_url,
        num_workers=args.num_workers,
        pooling=args.pooling,
        search_data_path=args.search_data_path,
        max_mz=args.max_mz,
        preprocessing=args.preprocessing,
        min_intensity=args.min_intensity,
    )


if __name__ == "__main__":
    main()
