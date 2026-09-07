"""Extract InstaNovo transformer encoder embeddings into the foundation model HDF5 format."""

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
from instanovo.transformer.model import InstaNovo
from instanovo_fm.utils.checkpoints import resolve_checkpoint
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


def extract(
    parquet_path: str,
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    checkpoint_url: Optional[str] = None,
    num_workers: int = 4,
    pooling: str = "latent",
    search_data_path: str = "data/search_data.xlsx",
    max_mz: float = 2500.0,
    min_intensity: Optional[float] = None,
) -> None:
    """Extract InstaNovo encoder embeddings and save in foundation-model HDF5 format.

    Args:
        parquet_path: Path to input parquet file(s).
        output_dir: Directory to write embeddings.h5 and index.faiss.
        checkpoint_path: Local path to instanovo checkpoint (.ckpt).
        batch_size: Inference batch size.
        device: PyTorch device string ("cuda", "cpu", "cuda:1", …).
        max_samples: Maximum number of spectra to embed.
        checkpoint_url: If provided and checkpoint_path doesn't exist, download from here.
        num_workers: DataLoader worker count.
        pooling: Pooling strategy — "latent" (the prepended latent spectrum token at
            position 1, InstaNovo's designed summary; default) or "mean_peaks" (mean over
            non-padding peak tokens at positions 2+, excluding the prepended precursor and
            latent tokens).
        search_data_path: Path to the search-data table for USI-based frag_type,
            search_instrument and search_detector lookup.
        max_mz: m/z normalisation divisor (the foundation model's ``max_mz``). The baseline
            scripts source this from the model config; defaults to 2500.0 for direct invocation.
        min_intensity: Optional override for the FoundationalDataProcessor intensity threshold
            (default 0.01 when None).
    """
    if pooling not in ("latent", "mean_peaks"):
        raise ValueError(f"pooling must be 'latent' or 'mean_peaks', got {pooling!r}")
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    # ------------------------------------------------------------------ #
    # 1. Resolve checkpoint                                                #
    # ------------------------------------------------------------------ #
    ckpt = resolve_checkpoint(
        checkpoint_path,
        checkpoint_url,
        not_found_hint="https://github.com/instadeepai/InstaNovo/releases/download/1.2.0/instanovo-v1.2.0.ckpt",
    )

    # ------------------------------------------------------------------ #
    # 2. Load model                                                        #
    # ------------------------------------------------------------------ #
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Loading InstaNovo checkpoint from {ckpt}")
    model, config = InstaNovo.load(str(ckpt))
    model = model.to(torch_device)
    model.eval()
    logger.info(f"Model loaded (dim_model={config.get('dim_model', '?')})")

    # ------------------------------------------------------------------ #
    # 3. Build DataLoader                                                  #
    # ------------------------------------------------------------------ #
    extra: dict[str, Any] = {} if min_intensity is None else {"min_intensity": min_intensity}
    if min_intensity is not None:
        logger.info(f"Foundation preprocessing with min_intensity override = {min_intensity}")
    dataloader = build_dataloader(parquet_path, batch_size, max_samples, num_workers, max_mz=max_mz, **extra)

    # ------------------------------------------------------------------ #
    # 4. Embedding loop (InstaNovo-specific encode + pool)                 #
    # ------------------------------------------------------------------ #
    all_embeddings: list = []
    all_metadata: list = []
    total_processed = 0

    with torch.inference_mode():
        for batch in dataloader:
            spectra = batch["spectra"].to(torch_device)  # (B, 200, 2)
            spectra_mask = batch["spectra_mask"].to(torch_device)  # (B, 200), True=padding
            precursors = batch["precursors"].to(torch_device)  # (B, 3): [mass, charge, mz]

            # Denormalise m/z: FoundationalDataProcessor divides by max_mz
            spectra_raw = spectra.clone()
            # TODO just don't normalise?
            spectra_raw[:, :, 0] = spectra_raw[:, :, 0] * max_mz

            max_charge = model.charge_encoder.num_embeddings
            precursors = precursors.clone()
            precursors[:, 1] = precursors[:, 1].clamp(min=1, max=max_charge)

            # Output layout: [precursor | latent | peak_1 ... peak_N]
            memories, _ = model._encoder(spectra_raw, precursors, spectra_mask)

            assert memories.shape[1] == spectra_raw.shape[1] + 2, (
                f"Unexpected InstaNovo encoder layout: {memories.shape[1]} tokens for {spectra_raw.shape[1]} peaks "
                "(expected n_peaks+2 — prepended precursor + latent tokens). InstaNovo encoder may have changed."
            )

            if pooling == "latent":
                embedding = memories[:, 1, :]  # (B, D)
            else:
                # Sequence layout: [precursor(0), latent(1), peak_1(2)...peak_N]
                # spectra_mask: (B, N_peaks), True=padding
                peak_tokens = memories[:, 2:, :]  # (B, N_peaks, D)
                # Expand mask to (B, N_peaks, 1), invert so True=valid
                valid = (~spectra_mask).unsqueeze(-1).float()  # (B, N_peaks, 1)
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
    """Extract InstaNovo embeddings."""
    parser = argparse.ArgumentParser(description="Extract InstaNovo encoder embeddings into foundation-model HDF5 format")
    parser.add_argument("--parquet_path", required=True, help="Path to input parquet file or glob pattern")
    parser.add_argument("--output_dir", required=True, help="Directory to write embeddings.h5 and index.faiss")
    parser.add_argument("--checkpoint_path", required=True, help="Local path to instanovo_extended.ckpt")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", default="cuda", help="PyTorch device string: cuda, cpu, cuda:1, etc. (default: cuda)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap on number of spectra to embed (default: all)")
    parser.add_argument(
        "--checkpoint_url",
        default=None,
        help=(
            "URL to download the checkpoint if --checkpoint_path does not exist. "
            "Default InstaNovo release: "
            "https://github.com/instadeepai/InstaNovo/releases/download/1.0.0/instanovo_extended.ckpt"
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument(
        "--pooling",
        default="latent",
        choices=["latent", "mean_peaks"],
        help="Pooling strategy: 'latent' (prepended latent token at position 1, default) or 'mean_peaks'",
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
        "this from the model config; default 2500.0 for direct invocation.",
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=None,
        help="Override the FoundationalDataProcessor intensity threshold (default 0.01)",
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
        min_intensity=args.min_intensity,
    )


if __name__ == "__main__":
    main()
