from instanovo_fm.common.dataset import DataProcessor
from instanovo_fm.common.scheduler import (
    CosineWarmupScheduler,
    FinetuneScheduler,
    WarmupScheduler,
)
from instanovo_fm.common.tracking import MLFlowTracker, create_tracker
from instanovo_fm.common.utils import Timer, TrainingState


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load heavy training modules to avoid importing neptune/tensorboard at startup."""
    if name == "AccelerateDeNovoTrainer":
        from instanovo_fm.common.trainer import AccelerateDeNovoTrainer

        return AccelerateDeNovoTrainer
    if name == "AccelerateDeNovoPredictor":
        from instanovo_fm.common.predictor import AccelerateDeNovoPredictor

        return AccelerateDeNovoPredictor
    raise AttributeError(f"module 'instanovo_fm.common' has no attribute {name!r}")


__all__ = [
    "DataProcessor",
    "AccelerateDeNovoTrainer",
    "AccelerateDeNovoPredictor",
    "FinetuneScheduler",
    "WarmupScheduler",
    "CosineWarmupScheduler",
    "TrainingState",
    "Timer",
    # Experiment tracking
    "MLFlowTracker",
    "create_tracker",
]
