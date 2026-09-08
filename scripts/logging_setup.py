"""Shared logging configuration for pipeline CLI scripts."""

from __future__ import annotations

import logging


def configure_script_logging(*, verbose: bool = False) -> None:
    """Configure root logging for a script CLI entry point.

    Call once at the start of each Typer command. Prefer this over
    import-time ``basicConfig`` so importing modules under pytest does not
    reset the root logger.

    Args:
        verbose: If True, set root level to DEBUG; otherwise INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
