"""Baseline trainers and predictors rebased onto the vendored `common/`.

`instanovo.transformer.train.TransformerTrainer` in the pinned release inherits
the *released* `AccelerateDeNovoTrainer`, so anything subclassing it bypasses
`instanovo_fm.common` and loses the internal additions -- most visibly the
`cosine_warmup_hold` scheduler every foundational config selects. This package
holds the same trainer with its base repointed at the vendored one.
"""

from instanovo_fm.baselines.transformer_train import TransformerTrainer


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load the predictor."""
    if name == "TransformerPredictor":
        from instanovo_fm.baselines.transformer_predict import TransformerPredictor

        return TransformerPredictor
    raise AttributeError(f"module 'instanovo_fm.baselines' has no attribute {name!r}")


__all__ = ["TransformerTrainer", "TransformerPredictor"]
