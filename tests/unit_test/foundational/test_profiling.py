"""Tests for the simplified training profiling system."""

import time
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from instanovo_fm.trainer.profiling import (
    ComponentTimer,
    TrainingProfiler,
    create_profiler_from_config,
    profile_component,
)


# ── ComponentTimer tests ──────────────────────────────────────────────────────


class TestComponentTimer:
    def test_basic_timing(self):
        timer = ComponentTimer("test")
        with timer:
            time.sleep(0.01)
        stats = timer.get_stats()
        assert stats["count"] == 1
        assert stats["mean"] > 0
        assert stats["total"] > 0

    def test_warmup_skips_first_n(self):
        timer = ComponentTimer("test", warmup_steps=3)
        # First 3 should be skipped
        for i in range(5):
            with timer:
                time.sleep(0.001)

        stats = timer.get_stats()
        assert stats["count"] == 5  # Total count includes warmup
        assert stats["warmup_skipped"] == 3
        assert len(timer.times) == 2  # Only 2 recorded (5 - 3 warmup)

    def test_multiple_entries(self):
        timer = ComponentTimer("test")
        for _ in range(5):
            with timer:
                time.sleep(0.001)
        stats = timer.get_stats()
        assert stats["count"] == 5
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_max_history(self):
        timer = ComponentTimer("test", max_history=3)
        for _ in range(10):
            with timer:
                pass
        assert len(timer.times) == 3
        assert timer.count == 10

    def test_empty_stats(self):
        timer = ComponentTimer("test")
        stats = timer.get_stats()
        assert stats["mean"] == 0.0
        assert stats["count"] == 0

    def test_percentiles(self):
        timer = ComponentTimer("test")
        for _ in range(100):
            with timer:
                time.sleep(0.001)
        stats = timer.get_stats()
        # P95 should be >= P50
        assert stats["p95"] >= stats["p50"]
        assert stats["p99"] >= stats["p95"]


# ── TrainingProfiler tests ────────────────────────────────────────────────────


class TestTrainingProfiler:
    def _make_profiler(self, **overrides):
        config = {"enabled": True, "use_component_timing": True, "cuda_sync": False}
        config.update(overrides)
        return TrainingProfiler(config, output_dir="/tmp/test_profiling")

    def test_canonical_component_acceptance(self):
        profiler = self._make_profiler()
        for name in TrainingProfiler.ALL_COMPONENTS:
            ctx = profiler.time_component(name)
            assert ctx is not None
            with ctx:
                pass
        assert set(profiler.component_timers.keys()) == set(TrainingProfiler.ALL_COMPONENTS)

    def test_unknown_component_rejection(self):
        profiler = self._make_profiler()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx = profiler.time_component("nonexistent_component")
            assert len(w) == 1
            assert "Unknown profiling component" in str(w[0].message)
        # Should not add a timer for unknown component
        assert "nonexistent_component" not in profiler.component_timers

    def test_disabled_mode(self):
        profiler = self._make_profiler(enabled=False)
        ctx = profiler.time_component("forward")
        # Should be nullcontext
        with ctx:
            pass
        assert len(profiler.component_timers) == 0

    def test_component_timing_disabled(self):
        profiler = self._make_profiler(use_component_timing=False)
        ctx = profiler.time_component("forward")
        with ctx:
            pass
        assert len(profiler.component_timers) == 0


class TestBatchSizeAndThroughput:
    def test_set_batch_size(self):
        profiler = TrainingProfiler({"enabled": True}, output_dir="/tmp/test_profiling")
        profiler.set_batch_size(512)
        assert profiler._batch_size == 512

    def test_record_step_time(self):
        profiler = TrainingProfiler({"enabled": True, "warmup_steps": 0}, output_dir="/tmp/test_profiling")
        profiler.record_step_time(0.2)
        profiler.record_step_time(0.21)
        assert len(profiler._step_times) == 2
        assert profiler._step_times[0] == pytest.approx(0.2)

    def test_record_step_time_with_warmup(self):
        profiler = TrainingProfiler({"enabled": True, "warmup_steps": 3}, output_dir="/tmp/test_profiling")
        # First 3 should be skipped
        for i in range(5):
            profiler.record_step_time(0.1 + i * 0.01)

        assert len(profiler._step_times) == 2  # Only last 2 recorded
        assert profiler._step_warmup_count == 3

    def test_get_last_step_times_empty(self):
        profiler = TrainingProfiler({"enabled": True}, output_dir="/tmp/test_profiling")
        result = profiler.get_last_step_times()
        assert result == {}

    def test_get_last_step_times_with_data(self):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True, "warmup_steps": 0},
            output_dir="/tmp/test_profiling",
        )
        # Record a component time
        with profiler.time_component("forward"):
            time.sleep(0.001)
        # Record a step time
        profiler.record_step_time(0.05)

        result = profiler.get_last_step_times()
        assert "forward" in result
        assert "step" in result
        assert result["step"] == pytest.approx(0.05)


# ── Report generation tests ──────────────────────────────────────────────────


class TestReportGeneration:
    def test_report_format(self, tmp_path):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True, "cuda_sync": False, "warmup_steps": 0},
            output_dir=str(tmp_path),
        )

        # Simulate profiling data
        for _ in range(10):
            with profiler.time_component("data_loading"):
                time.sleep(0.005)
            with profiler.time_component("forward"):
                time.sleep(0.002)
            with profiler.time_component("backward"):
                time.sleep(0.001)
            with profiler.time_component("optimizer"):
                time.sleep(0.0005)
            profiler.record_step_time(0.01)

        profiler.set_batch_size(1024)
        profiler._generate_report()

        report_file = tmp_path / "profiling_report.txt"
        assert report_file.exists()
        content = report_file.read_text()

        # Check structural elements
        assert "TRAINING STEP PROFILE (10 steps)" in content
        assert "Component" in content
        assert "Mean (ms)" in content
        assert "P95 (ms)" in content
        assert "% of step" in content
        assert "accounted" in content
        assert "unaccounted" in content
        assert "step (total)" in content
        assert "Throughput:" in content
        assert "steps/sec" in content
        assert "samples/sec" in content

    def test_report_with_percentages(self, tmp_path):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True, "cuda_sync": False},
            output_dir=str(tmp_path),
        )

        # forward accounts for 100% of step
        for _ in range(5):
            with profiler.time_component("forward"):
                time.sleep(0.01)
            profiler.record_step_time(0.01)

        profiler._generate_report()
        content = (tmp_path / "profiling_report.txt").read_text()
        # Forward should show ~100%
        assert "forward" in content

    def test_report_no_steps(self, tmp_path):
        """Report should still be generated when no steps are recorded."""
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True, "cuda_sync": False, "warmup_steps": 0},
            output_dir=str(tmp_path),
        )
        profiler._generate_report()
        report_file = tmp_path / "profiling_report.txt"
        assert report_file.exists()
        content = report_file.read_text()
        assert "TRAINING STEP PROFILE (0 steps)" in content

    def test_report_with_validation(self, tmp_path):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True, "cuda_sync": False},
            output_dir=str(tmp_path),
        )

        for _ in range(3):
            with profiler.time_component("val_forward"):
                time.sleep(0.001)
            with profiler.time_component("val_metrics"):
                time.sleep(0.001)

        profiler._generate_report()
        content = (tmp_path / "profiling_report.txt").read_text()
        assert "VALIDATION PROFILE" in content
        assert "val_forward" in content
        assert "val_metrics" in content


# ── profile_component context manager tests ──────────────────────────────────


class TestProfileComponent:
    def test_enabled_profiler(self):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True},
            output_dir="/tmp/test_profiling",
        )
        with profile_component(profiler, "forward"):
            time.sleep(0.001)
        assert "forward" in profiler.component_timers
        assert profiler.component_timers["forward"].count == 1

    def test_disabled_profiler(self):
        profiler = TrainingProfiler(
            {"enabled": False},
            output_dir="/tmp/test_profiling",
        )
        with profile_component(profiler, "forward"):
            pass
        assert len(profiler.component_timers) == 0

    def test_none_profiler(self):
        # Should not raise
        with profile_component(None, "forward"):
            pass

    def test_unknown_component_warns(self):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True},
            output_dir="/tmp/test_profiling",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with profile_component(profiler, "bogus"):
                pass
            assert any("Unknown profiling component" in str(x.message) for x in w)


# ── create_profiler_from_config tests ─────────────────────────────────────────


class TestCreateProfilerFromConfig:
    def test_from_omegaconf(self):
        config = OmegaConf.create({
            "profiling": {
                "enabled": True,
                "use_component_timing": True,
                "cuda_sync": False,
            }
        })
        profiler = create_profiler_from_config(config)
        assert profiler.enabled is True
        assert profiler.cuda_sync is False

    def test_missing_profiling_key(self):
        config = OmegaConf.create({})
        profiler = create_profiler_from_config(config)
        assert profiler.enabled is False

    def test_from_dict(self):
        config = {
            "profiling": {
                "enabled": True,
                "use_torch_profiler": True,
            }
        }
        profiler = create_profiler_from_config(config)
        assert profiler.enabled is True
        assert profiler.use_torch_profiler is True

    def test_non_dict_config(self):
        """Config without .get attribute should produce disabled profiler."""
        profiler = create_profiler_from_config(42)
        assert profiler.enabled is False


# ── Class constants tests ─────────────────────────────────────────────────────


class TestClassConstants:
    def test_training_components(self):
        assert TrainingProfiler.TRAINING_COMPONENTS == (
            "data_loading", "forward", "backward", "optimizer"
        )

    def test_validation_components(self):
        assert TrainingProfiler.VALIDATION_COMPONENTS == ("val_forward", "val_metrics")

    def test_step_component(self):
        assert TrainingProfiler.STEP_COMPONENT == "step"

    def test_all_components_is_union(self):
        expected = (
            TrainingProfiler.TRAINING_COMPONENTS
            + TrainingProfiler.VALIDATION_COMPONENTS
            + (TrainingProfiler.STEP_COMPONENT,)
        )
        assert TrainingProfiler.ALL_COMPONENTS == expected

    def test_exactly_seven_components(self):
        assert len(TrainingProfiler.ALL_COMPONENTS) == 7


# ── Timing summary tests ─────────────────────────────────────────────────────


class TestTimingSummary:
    def test_summary_returns_dict(self):
        profiler = TrainingProfiler(
            {"enabled": True, "use_component_timing": True},
            output_dir="/tmp/test_profiling",
        )
        with profiler.time_component("forward"):
            time.sleep(0.001)
        summary = profiler.get_timing_summary()
        assert "forward" in summary
        assert "mean_ms" in summary["forward"]
        assert "p95_ms" in summary["forward"]

    def test_summary_disabled(self):
        profiler = TrainingProfiler(
            {"enabled": False},
            output_dir="/tmp/test_profiling",
        )
        assert profiler.get_timing_summary() == {}
