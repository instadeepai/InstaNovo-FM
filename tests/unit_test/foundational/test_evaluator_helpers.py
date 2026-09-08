"""Tests for EmbeddingEvaluator helper methods.

Focuses on the pre-filter fast path and multi-split orchestration logic,
using mocks to avoid requiring a GPU or real checkpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from instanovo_fm.eval.embed_eval_tasks.linear_probe import LinearProbeTask


# ---------------------------------------------------------------------------
# Tests: _all_tasks_require_multi_split
# ---------------------------------------------------------------------------


class TestAllTasksRequireMultiSplit:
    """Unit tests for EmbeddingEvaluator._all_tasks_require_multi_split."""

    def _make_evaluator(self):
        """Create a minimal EmbeddingEvaluator with a dummy config."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.evaluator import EmbeddingEvaluator

        cfg = OmegaConf.create({
            "evaluation": {
                "output_dir": "/tmp/test_eval",
                "checkpoint_path": None,
            },
            "dataset": {},
            "residues": {"residues": {}},
            "num_workers": 0,
        })
        return EmbeddingEvaluator(cfg)

    def test_returns_false_for_empty_list(self):
        ev = self._make_evaluator()
        assert ev._all_tasks_require_multi_split([]) is False

    def test_returns_true_when_all_multi_split(self):
        ev = self._make_evaluator()
        # LinearProbeTask has requires_multi_split = True
        # Patch the registry to avoid needing all tasks registered
        with patch(
            "instanovo_fm.eval.evaluator.get_task",
            side_effect=lambda name: LinearProbeTask,
        ):
            assert ev._all_tasks_require_multi_split(["linearprobetask"]) is True
            assert ev._all_tasks_require_multi_split(["a", "b"]) is True

    def test_returns_false_when_any_not_multi_split(self):
        ev = self._make_evaluator()

        class NoMultiSplitTask:
            requires_multi_split = False

        def _mock_get_task(name):
            if name == "linearprobetask":
                return LinearProbeTask
            return NoMultiSplitTask

        with patch(
            "instanovo_fm.eval.evaluator.get_task",
            side_effect=_mock_get_task,
        ):
            assert ev._all_tasks_require_multi_split(["linearprobetask", "other"]) is False

    def test_returns_false_when_task_not_found(self):
        ev = self._make_evaluator()
        with patch(
            "instanovo_fm.eval.evaluator.get_task",
            side_effect=KeyError("not found"),
        ):
            assert ev._all_tasks_require_multi_split(["unknown_task"]) is False


# ---------------------------------------------------------------------------
# Tests: _generate_multi_split_embeddings — finally guard
# ---------------------------------------------------------------------------


class TestGenerateMultiSplitEmbeddingsFinally:
    """Verify that self.dataloader is restored even when generate_embeddings raises."""

    def _make_evaluator_with_model(self):
        from omegaconf import OmegaConf

        from instanovo_fm.eval.evaluator import EmbeddingEvaluator

        cfg = OmegaConf.create({
            "evaluation": {"output_dir": "/tmp/test_eval", "checkpoint_path": None},
            "dataset": {"train_path": "/fake/train", "valid_path": "/fake/valid", "test_path": "/fake/test"},
            "residues": {"residues": {}},
            "num_workers": 0,
        })
        ev = EmbeddingEvaluator(cfg)
        ev.model = MagicMock()
        ev.data_processor = MagicMock()
        return ev

    def test_dataloader_restored_on_exception(self):
        ev = self._make_evaluator_with_model()
        original_dl = MagicMock(name="original_dataloader")
        ev.dataloader = original_dl

        # setup_dataloader succeeds first time, then generate_embeddings raises
        ev.setup_dataloader = MagicMock(return_value=MagicMock(name="new_dl"))
        ev.generate_embeddings = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            ev._generate_multi_split_embeddings()

        # Despite the exception, dataloader must be restored
        assert ev.dataloader is original_dl

    def test_dataloader_restored_on_success(self):
        ev = self._make_evaluator_with_model()
        original_dl = MagicMock(name="original_dataloader")
        ev.dataloader = original_dl

        fake_emb = np.zeros((10, 16), dtype=np.float32)
        fake_meta = {"charge": np.ones(10)}

        ev.setup_dataloader = MagicMock(return_value=MagicMock())
        ev.generate_embeddings = MagicMock(return_value=(fake_emb, fake_meta, None))

        result = ev._generate_multi_split_embeddings()

        assert ev.dataloader is original_dl
        assert "train" in result or "valid" in result or "test" in result

    def test_precomputed_splits_reused(self):
        ev = self._make_evaluator_with_model()
        ev.dataloader = MagicMock()

        precomputed_emb = np.ones((5, 8), dtype=np.float32)
        precomputed_meta = {"charge": np.ones(5)}
        precomputed = {"train": (precomputed_emb, precomputed_meta)}

        fake_emb = np.zeros((10, 8), dtype=np.float32)
        fake_meta = {"charge": np.zeros(10)}
        ev.setup_dataloader = MagicMock(return_value=MagicMock())
        ev.generate_embeddings = MagicMock(return_value=(fake_emb, fake_meta, None))

        result = ev._generate_multi_split_embeddings(precomputed=precomputed)

        # Precomputed train split should be reused (generate_embeddings not called for it)
        assert np.array_equal(result["train"][0], precomputed_emb)
        # generate_embeddings should have been called for valid + test only (2 times)
        assert ev.generate_embeddings.call_count == 2


# ---------------------------------------------------------------------------
# Tests: run_evaluation_tasks — pre_filtered flag propagation
# ---------------------------------------------------------------------------


class TestRunEvaluationTasksPreFiltered:
    """Verify pre_filtered=True is passed through to task.run()."""

    def _make_evaluator(self):
        from omegaconf import OmegaConf

        from instanovo_fm.eval.evaluator import EmbeddingEvaluator

        cfg = OmegaConf.create({
            "evaluation": {
                "output_dir": "/tmp/test_eval",
                "checkpoint_path": None,
                "tasks_to_run": ["linearprobetask"],
                "task_configs": {
                    "linearprobetask": {
                        "targets": ["precursor_charge"],
                        "train_samples": 100,
                        "val_samples": 20,
                        "test_samples": 20,
                    }
                },
            },
            "dataset": {},
            "residues": {"residues": {}},
            "num_workers": 0,
        })
        ev = EmbeddingEvaluator(cfg)
        ev.model = MagicMock()
        ev.dataloader = MagicMock()
        return ev

    def test_pre_filtered_passed_to_multi_split_task(self):
        ev = self._make_evaluator()

        rng = np.random.RandomState(0)
        n, d = 50, 16

        def _make_split(n):
            emb = rng.randn(n, d).astype(np.float32)
            meta = {"precursor_charge": rng.randint(1, 4, size=n)}
            return emb, meta

        multi_splits = {
            "train": _make_split(100),
            "valid": _make_split(20),
            "test": _make_split(20),
        }

        captured_kwargs = {}

        class CapturingProbeTask(LinearProbeTask):
            def run(self, emb, meta, faiss_index, **kwargs):
                captured_kwargs.update(kwargs)
                return {"task_name": "Linear Probe", "mode": "pre_filtered_project_disjoint", "targets": {}, "split_info": {}, "config": {}}

        emb = np.zeros((1, d), dtype=np.float32)
        meta = {}

        with patch("instanovo_fm.eval.evaluator.get_task", return_value=CapturingProbeTask):
            ev.run_evaluation_tasks(
                embeddings=emb,
                metadata=meta,
                faiss_index=None,
                precomputed_splits=multi_splits,
                pre_filtered=True,
            )

        assert captured_kwargs.get("pre_filtered") is True
        assert "splits" in captured_kwargs
