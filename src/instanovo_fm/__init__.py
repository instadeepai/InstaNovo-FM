"""InstaNovo-FM: a self-supervised foundation model for tandem mass spectra."""

from omegaconf import OmegaConf

__version__ = "0.1.0.dev0"


def _upstream_config_dir() -> str:
    from instanovo_fm.utils.hydra_config import upstream_config_dir

    return str(upstream_config_dir())


# `configs/denovo.yaml` inherits configs that live in the installed `instanovo`
# package, so Hydra needs that directory on its search path. A `@hydra.main`
# entry point takes no overrides, so the path has to come from the config
# itself -- `file://${instanovo_configs:}` -- and it is only knowable at
# runtime. See instanovo_fm/utils/hydra_config.py for why `pkg://` cannot do
# this. Registered here because every entry point imports this package first.
OmegaConf.register_new_resolver("instanovo_configs", _upstream_config_dir, replace=True)
