"""Extract Casanovo spectrum encoder embeddings into the foundation model HDF5 format."""

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
from instanovo_fm.utils.checkpoints import resolve_checkpoint
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Casanovo's native spectrum preprocessing (casanovo config.yaml / dataloaders.py).
NATIVE_PREPROCESSING: dict[str, Any] = {
    "n_peaks": 150,
    "min_mz": 50.0,
    "max_mz": 2500.0,
    "min_intensity": 0.01,
    "remove_precursor_tol": 2.0,
}


def _require_casanovo() -> Any:
    """Import Spec2Pep from casanovo, raising a clear error if not installed."""
    try:
        from casanovo.denovo.model import Spec2Pep  # type: ignore[import]

        return Spec2Pep  # type: ignore[no-any-return]
    except ImportError as exc:
        raise ImportError("casanovo is not installed. Install it with:\n    uv pip install casanovo\nthen re-run this script.") from exc


def extract(
    parquet_path: str,
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    checkpoint_url: Optional[str] = None,
    num_workers: int = 4,
    pooling: str = "cls",
    search_data_path: str = "data/search_data.xlsx",
    max_mz: float = 2500.0,
    preprocessing: str = "foundation",
    min_intensity: Optional[float] = None,
) -> None:
    """Extract Casanovo encoder embeddings and save in foundation-model HDF5 format.

    Args:
        parquet_path: Path to input parquet file(s).
        output_dir: Directory to write embeddings.h5 and index.faiss.
        checkpoint_path: Local path to casanovo checkpoint (.ckpt).
        batch_size: Inference batch size.
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to embed.
        checkpoint_url: If provided and checkpoint_path doesn't exist, download from here.
        num_workers: DataLoader worker count.
        pooling: Pooling strategy — "cls" (global token at position 0, default) or
            "mean_peaks" (mean over non-padding peak tokens at positions 1+).
        search_data_path: Path to the search-data table for USI-based frag_type,
            search_instrument and search_detector lookup.
        max_mz: m/z normalisation divisor for the foundation preprocessing (the foundation model's
            ``max_mz``). Ignored when ``preprocessing="native"`` (native m/z range is used instead).
        preprocessing: "foundation" (shared FoundationalDataProcessor, 200 peaks, normalised m/z,
            default) or "native" (Casanovo's own preprocessing — see ``NATIVE_PREPROCESSING`` —
            with raw m/z).
        min_intensity: Optional override for the preprocessing intensity threshold. If None, the
            mode default is used (foundation: 0.01; native: ``NATIVE_PREPROCESSING['min_intensity']``).

    Raises:
        ValueError: If ``pooling`` or ``preprocessing`` is not a recognised value.
        ImportError: If faiss is not available.
        FileNotFoundError: If the checkpoint is missing and no ``checkpoint_url`` is given.
    """
    if pooling not in ("cls", "mean_peaks"):
        raise ValueError(f"pooling must be 'cls' or 'mean_peaks', got {pooling!r}")
    if preprocessing not in ("foundation", "native"):
        raise ValueError(f"preprocessing must be 'foundation' or 'native', got {preprocessing!r}")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    spec2pep_cls = _require_casanovo()

    # ------------------------------------------------------------------ #
    # 1. Resolve checkpoint                                                #
    # ------------------------------------------------------------------ #
    ckpt = resolve_checkpoint(
        checkpoint_path,
        checkpoint_url,
        not_found_hint="  https://github.com/Noble-Lab/casanovo/releases",
    )

    # ------------------------------------------------------------------ #
    # 2. Load model                                                        #
    # ------------------------------------------------------------------ #
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading Casanovo checkpoint from {ckpt}")
    model = spec2pep_cls.load_from_checkpoint(str(ckpt), map_location="cpu", weights_only=False)
    model = model.to(torch_device)
    model.eval()
    logger.info("Casanovo model loaded")

    # ------------------------------------------------------------------ #
    # 3. Build DataLoader                                                  #
    # ------------------------------------------------------------------ #
    if preprocessing == "native":
        native_params = dict(NATIVE_PREPROCESSING)
        if min_intensity is not None:
            native_params["min_intensity"] = min_intensity
        logger.info(f"Using Casanovo-native preprocessing (raw m/z): {native_params}")
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
    # 4. Embedding loop (Casanovo-specific encode + pool)                  #
    # ------------------------------------------------------------------ #
    all_embeddings: list = []
    all_metadata: list = []
    total_processed = 0

    with torch.inference_mode():
        for batch in dataloader:
            spectra = batch["spectra"].to(torch_device)  # (B, n_peaks, 2)
            spectra_raw = spectra.clone()
            if preprocessing == "foundation":
                spectra_raw[:, :, 0] = spectra_raw[:, :, 0] * max_mz

            # Split into separate m/z and intensity tensors expected by Casanovo encoder.
            mzs = spectra_raw[:, :, 0]  # (B, L)
            intensities = spectra_raw[:, :, 1]  # (B, L)

            # Output layout: [cls(0) | peak_1(1) ... peak_N]
            memories, _ = model.encoder(mzs, intensities)

            assert memories.shape[1] == mzs.shape[1] + 1, (
                f"Unexpected Casanovo encoder layout: {memories.shape[1]} tokens for {mzs.shape[1]} peaks "
                "(expected n_peaks+1 — one prepended global token). casanovo/depthcharge version may have changed."
            )

            if pooling == "cls":
                embedding = memories[:, 0, :]  # (B, D)
            else:
                peak_tokens = memories[:, 1:, :]  # (B, N_peaks, D)
                valid = (mzs > 0).unsqueeze(-1).float()  # (B, N_peaks, 1)
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
    """Extract Casanovo embeddings."""
    parser = argparse.ArgumentParser(description="Extract Casanovo encoder embeddings into foundation-model HDF5 format")
    parser.add_argument("--parquet_path", required=True, help="Path to input parquet file or glob pattern")
    parser.add_argument("--output_dir", required=True, help="Directory to write embeddings.h5 and index.faiss")
    parser.add_argument("--checkpoint_path", required=True, help="Local path to casanovo checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", default="cuda", help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on number of spectra to embed (default: all)")
    parser.add_argument(
        "--checkpoint_url",
        default=None,
        help=("URL to download the checkpoint if --checkpoint_path does not exist. Find releases at: https://github.com/Noble-Lab/casanovo/releases"),
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument(
        "--pooling", default="cls", choices=["cls", "mean_peaks"], help="Pooling strategy: 'cls' (global token, default) or 'mean_peaks'"
    )
    parser.add_argument(
        "--search_data_path",
        default="data/search_data.xlsx",
        help="Path to the search-data table for USI-based frag_type/search_instrument lookup",
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
        "'native' (Casanovo's own 150-peak / 50-2500 m/z / precursor-removal pipeline)",
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=None,
        help="Override the preprocessing intensity threshold (default: mode default — foundation 0.01, native 0.01)",
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
