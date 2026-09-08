"""Performance profiling utilities for foundational model training.

Provides simple, non-overlapping profiling with actionable output:
percentage-of-step, throughput, and unaccounted time.

Training components (4 + total):
    data_loading  — next(dataloader_iter)
    forward       — model forward + all loss computation
    backward      — accelerator.backward(loss)
    optimizer     — grad_clip + optimizer.step + scheduler.step + zero_grad
    step          — wall-clock between consecutive step completions

Validation components (2):
    val_forward   — forward(batch, return_preds=True)
    val_metrics   — all metrics updates + gather + final compute
"""

import contextlib
import json
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch
import torch.profiler

from instanovo import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class ComponentTimer:
    """High-precision timer for measuring component execution times.

    Supports warmup period to exclude first N measurements from statistics
    (to avoid first-step JIT/CUDA compilation distorting means/percentiles).
    """

    def __init__(self, name: str, max_history: int = 1000, warmup_steps: int = 0) -> None:
        """Initialise the input."""
        self.name = name
        self.max_history = max_history
        self.warmup_steps = warmup_steps
        self.times: deque[float] = deque(maxlen=max_history)
        self.start_time: Optional[float] = None
        self.total_time = 0.0
        self.count = 0
        self._warmup_count = 0  # Separate counter for warmup steps

    def __enter__(self) -> Any:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time

            # Always increment total count (includes warmup)
            self.count += 1

            # Skip warmup steps from statistics
            if self._warmup_count < self.warmup_steps:
                self._warmup_count += 1
            else:
                # Only record post-warmup timings
                self.times.append(elapsed)
                self.total_time += elapsed

            self.start_time = None

    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics for this component (excluding warmup)."""
        if not self.times:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "total": 0.0,
                "count": self.count,
                "warmup_skipped": self._warmup_count,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        times_array = np.array(self.times)
        return {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "total": self.total_time,
            "count": self.count,  # Total count (including warmup)
            "warmup_skipped": self._warmup_count,
            "p50": float(np.percentile(times_array, 50)),
            "p95": float(np.percentile(times_array, 95)),
            "p99": float(np.percentile(times_array, 99)),
        }


class MemoryProfiler:
    """Memory usage profiler for GPU and CPU memory."""

    def __init__(self, track_gpu: bool = True, track_cpu: bool = True) -> None:
        """Initialise the input."""
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_cpu = track_cpu
        self.reset()

    def reset(self) -> None:
        """Reset memory tracking."""
        self.gpu_memory_history: list[Any] = []
        self.cpu_memory_history: list[Any] = []
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self, label: str = "") -> Any:
        """Take a memory snapshot."""
        snapshot: dict[str, Any] = {"label": label, "timestamp": time.time()}

        if self.track_gpu:
            snapshot.update(
                {
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "gpu_max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                }
            )
            self.gpu_memory_history.append(snapshot.copy())

        if self.track_cpu:
            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot.update(
                {
                    "cpu_rss_mb": memory_info.rss / 1024**2,
                    "cpu_vms_mb": memory_info.vms / 1024**2,
                    "cpu_percent": process.memory_percent(),
                }
            )
            self.cpu_memory_history.append(snapshot.copy())

        return snapshot

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage statistics."""
        stats = {}

        if self.track_gpu and self.gpu_memory_history:
            gpu_allocated = [s["gpu_allocated_mb"] for s in self.gpu_memory_history]
            gpu_reserved = [s["gpu_reserved_mb"] for s in self.gpu_memory_history]
            stats.update(
                {
                    "peak_gpu_allocated_mb": max(gpu_allocated),
                    "peak_gpu_reserved_mb": max(gpu_reserved),
                    "current_gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "current_gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                }
            )

        if self.track_cpu and self.cpu_memory_history:
            cpu_rss = [s["cpu_rss_mb"] for s in self.cpu_memory_history]
            cpu_percent = [s["cpu_percent"] for s in self.cpu_memory_history]
            stats.update(
                {
                    "peak_cpu_rss_mb": max(cpu_rss),
                    "peak_cpu_percent": max(cpu_percent),
                }
            )

        return stats


class TrainingProfiler:
    """Training profiler with non-overlapping, canonical component timing.

    Profiles exactly 7 components:
    - Training: data_loading, forward, backward, optimizer
    - Validation: val_forward, val_metrics
    - Step total: step (wall-clock between consecutive completions)
    """

    TRAINING_COMPONENTS = ("data_loading", "forward", "backward", "optimizer")
    VALIDATION_COMPONENTS = ("val_forward", "val_metrics")
    STEP_COMPONENT = "step"
    ALL_COMPONENTS = TRAINING_COMPONENTS + VALIDATION_COMPONENTS + (STEP_COMPONENT,)

    def __init__(self, config: Dict[str, Any], output_dir: str = "./profiling_results") -> None:
        """Initialise the input."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Profiling configuration
        self.enabled = config.get("enabled", False)
        self.use_torch_profiler = config.get("use_torch_profiler", False)
        self.use_component_timing = config.get("use_component_timing", True)
        self.use_memory_profiling = config.get("use_memory_profiling", False)
        self.cuda_sync = config.get("cuda_sync", True)
        self.tensorboard_profiling_interval = config.get("tensorboard_profiling_interval", 0)
        self.warmup_steps = config.get("warmup_steps", 5)  # Skip first N steps from stats

        # Torch profiler configuration
        self.torch_profiler_config = config.get("torch_profiler", {})
        self.profiler_schedule = self.torch_profiler_config.get("schedule", {"wait": 1, "warmup": 1, "active": 3, "repeat": 2})

        # Component timing
        self.component_timers: Dict[str, ComponentTimer] = {}
        self.memory_profiler = MemoryProfiler() if self.use_memory_profiling else None

        # Torch profiler instance
        self.torch_profiler = None
        self.profiler_step_count = 0

        # Step timing (manual, to avoid StopIteration issues with context managers)
        self._step_times: deque[float] = deque(maxlen=1000)
        self._step_warmup_count = 0  # Track warmup for manual step timing

        # Throughput tracking
        self._batch_size: Optional[int] = None

        # Results storage
        self.results: Dict[str, Any] = {
            "component_timing": {},
            "memory_usage": {},
            "torch_profiler_traces": [],
        }

        if self.enabled:
            logger.info(f"Training profiler enabled. Results will be saved to: {self.output_dir}")

    def set_batch_size(self, batch_size: int) -> None:
        """Set effective batch size for throughput computation."""
        self._batch_size = batch_size

    def time_component(self, name: str) -> contextlib.AbstractContextManager:
        """Get a context manager for timing a canonical component.

        Validates that the name is in the canonical component set.
        Returns nullcontext() for unknown names (with a warning) or when disabled.
        """
        if not self.enabled or not self.use_component_timing:
            return contextlib.nullcontext()

        if name not in self.ALL_COMPONENTS:
            warnings.warn(
                f"Unknown profiling component '{name}'. Allowed: {self.ALL_COMPONENTS}. Returning nullcontext().",
                stacklevel=2,
            )
            return contextlib.nullcontext()

        if name not in self.component_timers:
            self.component_timers[name] = ComponentTimer(name, warmup_steps=self.warmup_steps)
        return self.component_timers[name]

    def get_component_timer(self, name: str) -> Any:
        """Get or create a component timer, routing through validation."""
        return self.time_component(name)

    def record_step_time(self, elapsed: float) -> None:
        """Record a step's wall-clock time (manual, avoids StopIteration issues).

        Skips warmup steps to avoid first-step JIT/CUDA compilation distorting stats.
        """
        if self._step_warmup_count < self.warmup_steps:
            self._step_warmup_count += 1
        else:
            self._step_times.append(elapsed)

    def get_last_step_times(self) -> Dict[str, float]:
        """Get timing data from the most recent step for optional TensorBoard logging.

        Returns a dict mapping component names to their last recorded time in seconds.
        """
        result = {}
        for name, timer in self.component_timers.items():
            if timer.times:
                result[name] = timer.times[-1]
        if self._step_times:
            result["step"] = self._step_times[-1]
        return result

    def setup_torch_profiler(self) -> None:
        """Setup PyTorch profiler with appropriate configuration."""
        if not (self.enabled and self.use_torch_profiler):
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=self.profiler_schedule["wait"],
            warmup=self.profiler_schedule["warmup"],
            active=self.profiler_schedule["active"],
            repeat=self.profiler_schedule["repeat"],
        )

        self.torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.torch_profiler_config.get("record_shapes", True),
            profile_memory=self.torch_profiler_config.get("profile_memory", True),
            with_stack=self.torch_profiler_config.get("with_stack", False),
            with_flops=self.torch_profiler_config.get("with_flops", False),
            with_modules=self.torch_profiler_config.get("with_modules", False),
        )

        logger.info("PyTorch profiler configured and ready")

    def _on_trace_ready(self, prof: Any) -> None:
        """Callback when PyTorch profiler trace is ready."""
        try:
            trace_file = self.output_dir / f"torch_trace_step_{self.profiler_step_count}.json"
            prof.export_chrome_trace(str(trace_file))

            stats_file = self.output_dir / f"torch_stats_step_{self.profiler_step_count}.txt"
            key_averages = prof.key_averages()
            with open(stats_file, "w") as f:
                f.write(key_averages.table(sort_by="cuda_time_total", row_limit=50))

            self.results["torch_profiler_traces"].append(
                {
                    "step": self.profiler_step_count,
                    "trace_file": str(trace_file),
                    "stats_file": str(stats_file),
                }
            )

            logger.info(f"PyTorch profiler trace saved: {trace_file}")

        except Exception as e:
            logger.warning(f"Failed to save PyTorch profiler trace: {e}")

    def start_profiling(self) -> None:
        """Start profiling session."""
        if not self.enabled:
            return

        self.setup_torch_profiler()
        if self.torch_profiler:
            self.torch_profiler.start()

        if self.memory_profiler:
            self.memory_profiler.reset()
            self.memory_profiler.snapshot("profiling_start")

        logger.info("Profiling session started")

    def step(self) -> None:
        """Step the profiler (call this each training step)."""
        if not self.enabled:
            return

        if self.torch_profiler:
            self.torch_profiler.step()
            self.profiler_step_count += 1

        if self.memory_profiler:
            self.memory_profiler.snapshot(f"step_{self.profiler_step_count}")

    def stop_profiling(self) -> None:
        """Stop profiling and save results."""
        if not self.enabled:
            return

        if self.torch_profiler:
            self.torch_profiler.stop()

        if self.memory_profiler:
            self.memory_profiler.snapshot("profiling_end")
            self.results["memory_usage"] = self.memory_profiler.get_peak_usage()

        # Collect component timing results
        for name, timer in self.component_timers.items():
            self.results["component_timing"][name] = timer.get_stats()

        # Save comprehensive results
        self._save_results()
        logger.info("Profiling session completed and results saved")

    def _save_results(self) -> None:
        """Save profiling results to disk."""
        try:
            summary_file = self.output_dir / "profiling_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            if self.results["component_timing"]:
                timing_file = self.output_dir / "component_timing_detailed.json"
                with open(timing_file, "w") as f:
                    json.dump(self.results["component_timing"], f, indent=2)

            self._generate_report()

            logger.info(f"Profiling results saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save profiling results: {e}")

    def _generate_report(self) -> None:
        """Generate a human-readable profiling report with percentages and throughput."""
        report_file = self.output_dir / "profiling_report.txt"

        # Compute step stats
        step_times = np.array(self._step_times) if self._step_times else None
        step_count = len(self._step_times) if self._step_times else 0
        step_mean = float(np.mean(step_times)) if step_times is not None and len(step_times) > 0 else 0.0
        step_p95 = float(np.percentile(step_times, 95)) if step_times is not None and len(step_times) > 0 else 0.0

        with open(report_file, "w") as f:
            # Training step profile
            if self.warmup_steps > 0:
                f.write(f"TRAINING STEP PROFILE ({step_count} steps, {self._step_warmup_count} warmup skipped)\n")
            else:
                f.write(f"TRAINING STEP PROFILE ({step_count} steps)\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Component':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'% of step':<10}\n")
            f.write("-" * 60 + "\n")

            accounted = 0.0
            for comp in self.TRAINING_COMPONENTS:
                if comp in self.component_timers:
                    stats = self.component_timers[comp].get_stats()
                    mean_ms = stats["mean"] * 1000
                    p95_ms = stats["p95"] * 1000
                    pct = (stats["mean"] / step_mean * 100) if step_mean > 0 else 0.0
                    accounted += stats["mean"]
                    f.write(f"{comp:<20} {mean_ms:<12.2f} {p95_ms:<12.2f} {pct:<9.1f}%\n")

            f.write("-" * 60 + "\n")

            # Accounted / unaccounted
            if step_mean > 0:
                accounted_ms = accounted * 1000
                unaccounted = step_mean - accounted
                unaccounted_ms = unaccounted * 1000
                accounted_pct = accounted / step_mean * 100
                unaccounted_pct = unaccounted / step_mean * 100
                f.write(f"{'accounted':<20} {accounted_ms:<12.2f} {'':<12} {accounted_pct:<9.1f}%\n")
                f.write(f"{'unaccounted':<20} {unaccounted_ms:<12.2f} {'':<12} {unaccounted_pct:<9.1f}%\n")

            step_mean_ms = step_mean * 1000
            step_p95_ms = step_p95 * 1000
            f.write(f"{'step (total)':<20} {step_mean_ms:<12.2f} {step_p95_ms:<12.2f} {'100.0':<9}%\n")
            f.write("\n")

            # Throughput
            if step_mean > 0:
                steps_per_sec = 1.0 / step_mean
                f.write(f"Throughput: {steps_per_sec:.1f} steps/sec")
                if self._batch_size is not None:
                    samples_per_sec = steps_per_sec * self._batch_size
                    f.write(f" | {samples_per_sec:.0f} samples/sec")
                f.write("\n")

            # GPU memory snapshot
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                pct = allocated_gb / total_gb * 100 if total_gb > 0 else 0
                f.write(f"GPU Memory: {allocated_gb:.1f} GB allocated / {total_gb:.1f} GB total ({pct:.1f}%)\n")

            f.write("\n")

            # Validation profile (if available)
            val_components = [c for c in self.VALIDATION_COMPONENTS if c in self.component_timers]
            if val_components:
                f.write("VALIDATION PROFILE\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Component':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Count':<8}\n")
                f.write("-" * 60 + "\n")
                for comp in val_components:
                    stats = self.component_timers[comp].get_stats()
                    mean_ms = stats["mean"] * 1000
                    p95_ms = stats["p95"] * 1000
                    f.write(f"{comp:<20} {mean_ms:<12.2f} {p95_ms:<12.2f} {stats['count']:<8}\n")
                f.write("\n")

            # Memory usage summary
            if self.results["memory_usage"]:
                f.write("MEMORY USAGE SUMMARY\n")
                f.write("-" * 30 + "\n")
                for key, value in self.results["memory_usage"].items():
                    f.write(f"{key}: {value:.2f}\n")
                f.write("\n")

            # PyTorch profiler traces
            if self.results["torch_profiler_traces"]:
                f.write("PYTORCH PROFILER TRACES\n")
                f.write("-" * 30 + "\n")
                for trace in self.results["torch_profiler_traces"]:
                    f.write(f"Step {trace['step']}: {trace['trace_file']}\n")
                f.write("\nView traces in Chrome by opening chrome://tracing/ and loading the .json files\n")

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of current timing statistics."""
        if not (self.enabled and self.use_component_timing):
            return {}

        summary = {}
        for name, timer in self.component_timers.items():
            stats = timer.get_stats()
            summary[name] = {
                "mean_ms": stats["mean"] * 1000,
                "total_s": stats["total"],
                "count": stats["count"],
                "p95_ms": stats["p95"] * 1000,
            }

        return summary


# Context manager for easy profiling
@contextlib.contextmanager  # type: ignore[arg-type]
def profile_component(profiler: Optional[TrainingProfiler], component_name: str) -> None:  # type: ignore[misc]
    """Context manager for profiling a specific component.

    Routes through TrainingProfiler.time_component() for validation.
    """
    if profiler is not None and profiler.enabled and profiler.use_component_timing:
        with profiler.time_component(component_name):
            yield
    else:
        yield


def create_profiler_from_config(config: Any, output_dir: str = "./profiling_results") -> TrainingProfiler:
    """Create a profiler instance from configuration."""
    profiling_config = config.get("profiling", {}) if hasattr(config, "get") else {}
    return TrainingProfiler(profiling_config, output_dir)
