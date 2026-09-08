"""Checkpoint resolution for the external-baseline runners.

Vendored from the internal repo's ``instanovo/utils/checkpoints.py``. The
published ``instanovo`` package does not expose this module, and
``eval/run_xuanjinovo.py`` and ``eval/extract_xuanjinovo_embeddings.py`` need
``resolve_checkpoint`` to fetch a checkpoint from a URL or an S3 path.
Delete this copy if it is ever exported upstream.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

S3_NOT_CONFIGURED_HINT = "Set AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (e.g. via .env)."


def download_checkpoint(url: str, dest: Path) -> None:
    """Download a checkpoint file to ``dest`` (creating parent dirs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from {url} → {dest}")
    urllib.request.urlretrieve(url, str(dest))
    logger.info("Download complete")


def resolve_checkpoint(
    checkpoint_path: str,
    checkpoint_url: Optional[str] = None,
    *,
    not_found_hint: str = "",
    cache_dir: Path = Path("checkpoints"),
) -> Path:
    """Resolve a checkpoint that may live on S3, on the local filesystem, or behind a download URL.

    Resolution order:
        1. ``s3://...`` URI  -> downloaded to ``cache_dir/<basename>`` via ``S3FileHandler``
           (cached; the download is skipped if the local copy already exists).
        2. existing local file -> returned as-is.
        3. ``checkpoint_url`` provided -> downloaded via :func:`download_checkpoint`.
        4. otherwise -> ``FileNotFoundError``.

    Args:
        checkpoint_path: Local path or ``s3://bucket/key`` URI to the checkpoint.
        checkpoint_url: Optional HTTP(S) URL to download from if ``checkpoint_path`` is a
            non-existent local path.
        not_found_hint: Extra message appended to the ``FileNotFoundError`` (e.g. a manual
            download link) when nothing can be resolved.
        cache_dir: Directory S3 checkpoints are downloaded into (and cached across runs).

    Returns:
        Local path to the resolved checkpoint file.

    Raises:
        RuntimeError: If an ``s3://`` URI is given but S3 is not configured in the environment.
        FileNotFoundError: If the checkpoint cannot be resolved locally or downloaded.
    """
    if checkpoint_path.startswith("s3://"):
        if not S3FileHandler.s3_enabled():
            raise RuntimeError(f"Checkpoint '{checkpoint_path}' is an S3 URI but S3 is not configured. {S3_NOT_CONFIGURED_HINT}")
        dest = cache_dir / Path(checkpoint_path).name
        if dest.exists():
            logger.info(f"Using cached S3 checkpoint at {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading checkpoint from {checkpoint_path} → {dest}")
            S3FileHandler().download(checkpoint_path, str(dest))
            logger.info("Download complete")
        return dest

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        return ckpt
    if checkpoint_url:
        download_checkpoint(checkpoint_url, ckpt)
        return ckpt
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt}\n"
        "Pass --checkpoint_url to download automatically, an s3:// URI to --checkpoint_path, "
        f"or download manually.{f'{chr(10)}{not_found_hint}' if not_found_hint else ''}"
    )
