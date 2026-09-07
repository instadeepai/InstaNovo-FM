"""Hydra composition for this package's configs.

Two things have to be right for a config here to compose, and neither is the
default.

**This package's configs, not the dependency's.** ``instanovo.constants``
exports ``DEFAULT_TRAIN_CONFIG_PATH = "../configs"``, a path relative to
``instanovo/utils/cli_utils.py`` inside the installed package, so
``compose_config`` with that default looks in ``instanovo/configs/`` and cannot
see ``foundational.yaml`` or ``denovo.yaml``. ``FM_CONFIG_DIR`` is an absolute
path to this package's own ``configs/``.

**The dependency's configs on the search path.** ``configs/denovo.yaml``
inherits ``instanovo``, ``model: instanovo_base`` and ``dataset: default``,
which are the dependency's own configs. In the internal repository that file
sits inside ``instanovo/configs/`` next to them and they resolve locally; here
it has to reach across a package boundary. ``pkg://instanovo.configs`` does not
work for this -- the published package ships ``configs/`` as a data directory
with no ``__init__.py``, so Hydra's ``pkg://`` provider cannot see it -- and
copying the three parents in would duplicate the dependency's configuration.
So the installed directory goes on Hydra's search path instead.

For ``@hydra.main`` entry points, which take no overrides, the same path is
available in a config as ``file://${instanovo_configs:}`` via the resolver
registered in ``instanovo_fm/__init__.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

FM_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def upstream_config_dir() -> Path:
    """Absolute path to the installed ``instanovo`` package's ``configs/``."""
    import instanovo

    return Path(instanovo.__file__).resolve().parent / "configs"


def compose_fm_config(
    config_name: str,
    overrides: Optional[List[str]] = None,
    config_dir: Optional[str] = None,
) -> DictConfig:
    """Compose ``config_name`` from this package's configs.

    Args:
        config_name: Config to compose, e.g. ``foundational`` or ``denovo``.
        overrides: Hydra override strings.
        config_dir: Directory to compose from. Defaults to this package's
            ``configs/``; pass an absolute path to compose from elsewhere.

    Returns:
        The composed configuration.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    resolved = [] if overrides is None else list(overrides)
    # Leave a caller-supplied search path alone: overriding it twice is an error,
    # and someone passing their own knows where their configs are.
    if not any(o.startswith("hydra.searchpath") for o in resolved):
        resolved.append(f"hydra.searchpath=[file://{upstream_config_dir()}]")

    directory = str(Path(config_dir).resolve()) if config_dir else str(FM_CONFIG_DIR)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=directory, version_base=None):
        return compose(config_name=config_name, overrides=resolved)
