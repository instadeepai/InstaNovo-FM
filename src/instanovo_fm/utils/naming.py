"""Filename helpers.

``sanitize_filename`` is vendored from the internal repo's ``instanovo/utils/s3.py``.
The published ``instanovo`` package exports ``S3FileHandler`` and
``register_filesystem`` from that module but not this helper, and the data-analysis
figure writers need it to derive per-spectrum filenames from peptide sequences.
It is pure stdlib, so vendoring costs nothing; if it is ever exported upstream,
delete this module and import from ``instanovo.utils.s3`` instead.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_filename"]


def sanitize_filename(name: str, max_len: int = 128) -> str:
    """Make an arbitrary string safe to use as a filename component.

    Replaces any run of characters that are not alphanumerics, dot, underscore
    or hyphen with a single underscore, and trims leading/trailing separators.

    Args:
        name: The string to sanitise.
        max_len: Truncate the result to this many characters.

    Returns:
        A filename-safe string, never empty (falls back to ``"unnamed"``).
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    return (cleaned or "unnamed")[:max_len]
