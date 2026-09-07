"""Baseline trainers rebased onto the vendored `common/`.

`instanovo.transformer.train.TransformerTrainer` in the pinned release inherits
the *released* `AccelerateDeNovoTrainer`, so anything subclassing it bypasses
`instanovo_fm.common` and loses the internal additions -- most visibly the
`cosine_warmup_hold` scheduler every foundational config selects. This package
holds the same trainer with its base repointed at the vendored one.
"""

from instanovo_fm.baselines.transformer_train import TransformerTrainer

__all__ = ["TransformerTrainer"]
