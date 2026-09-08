"""Resolve the HuggingFace token from the environment.

Defined in one place because three entry points need it -- the release probe, the
tier upload, and the peptide-registry publish in ``scripts/splitting`` -- and a
token that works for one but not the others is a confusing failure.

Two names are accepted. ``INSTANOVO_FM_HF_TOKEN`` is preferred and is the one to
set; ``INSTANOVO_HF_TOKEN`` is still read so existing deployments keep working.
The resolver reports which name it used, because "the token is wrong" and "the
token I set is not the one being read" look identical from a 404.
"""

from __future__ import annotations

import os

TOKEN_ENV_VARS = ("INSTANOVO_FM_HF_TOKEN", "INSTANOVO_HF_TOKEN")


def resolve_hf_token() -> tuple[str, str]:
    """Return ``(token, env_var_name)``, preferring the FM-specific variable.

    Raises:
        RuntimeError: if neither variable is set, or the one that is set is empty.
    """
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value, name
    set_but_empty = [n for n in TOKEN_ENV_VARS if n in os.environ]
    if set_but_empty:
        raise RuntimeError(f"{', '.join(set_but_empty)} is set but empty")
    raise RuntimeError(f"none of {', '.join(TOKEN_ENV_VARS)} is set in the environment")
