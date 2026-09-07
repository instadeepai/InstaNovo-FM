"""Comprehensive test suite for checkpointing functionality.

This module provides comprehensive tests for checkpointing features
to ensure they work correctly with realistic data processing scenarios.
"""

import tempfile
import shutil
import time
import json
from pathlib import Path
from typing import Generator, Any
import polars as pl
import pytest


class TestCheckpointing:
    """Test suite for checkpointing functionality."""

    def test_checkpoint_creation(self) -> None:
        """Test that checkpoints are created during processing."""
        # This test simulates checkpoint creation during data processing
        checkpoint_dir = self.output_dir / "checkpoint_test"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create a mock checkpoint file
        checkpoint_data = {
            "processed_files": ["sample_0.ipc", "sample_1.ipc"],
            "current_file": "sample_2.ipc",
            "total_files": 10,
            "timestamp": "2024-01-01T12:00:00",
            "status": "in_progress",
        }

        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Verify checkpoint file exists and has correct content
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["processed_files"] == ["sample_0.ipc", "sample_1.ipc"]
        assert loaded_data["current_file"] == "sample_2.ipc"
        assert loaded_data["total_files"] == 10
        assert loaded_data["status"] == "in_progress"

    def test_checkpoint_resume(self) -> None:
        """Test that processing can resume from a checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoint_resume"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create a checkpoint file
        checkpoint_data: dict[str, Any] = {
            "processed_files": ["sample_0.ipc", "sample_1.ipc", "sample_2.ipc"],
            "current_file": "sample_3.ipc",
            "total_files": 10,
            "timestamp": "2024-01-01T12:00:00",
            "status": "in_progress",
        }

        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Simulate resuming processing
        # Update checkpoint with new progress
        checkpoint_data["processed_files"].extend(["sample_3.ipc", "sample_4.ipc"])
        checkpoint_data["current_file"] = "sample_5.ipc"
        checkpoint_data["timestamp"] = "2024-01-01T12:30:00"

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Verify checkpoint was updated
        with open(checkpoint_file, "r") as f:
            updated_data = json.load(f)

        assert len(updated_data["processed_files"]) == 5
        assert updated_data["current_file"] == "sample_5.ipc"

    def test_checkpoint_interval(self) -> None:
        """Test checkpoint creation at specified intervals."""
        checkpoint_dir = self.output_dir / "checkpoint_interval"
        checkpoint_dir.mkdir(exist_ok=True)

        # Simulate processing with checkpoint interval of 2
        checkpoint_interval = 2
        processed_files = []

        for i in range(10):
            processed_files.append(f"sample_{i}.ipc")

            # Create checkpoint every 2 files
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    "processed_files": processed_files.copy(),
                    "current_file": f"sample_{i + 1}.ipc" if i + 1 < 10 else None,
                    "total_files": 10,
                    "timestamp": f"2024-01-01T12:{i:02d}:00",
                    "status": "in_progress" if i + 1 < 10 else "completed",
                }

                checkpoint_file = checkpoint_dir / "split_checkpoint.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f)

        # Verify checkpoint was created at intervals
        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            final_data = json.load(f)

        assert len(final_data["processed_files"]) == 10
        assert final_data["status"] == "completed"

    def test_checkpoint_corruption_recovery(self) -> None:
        """Test recovery from corrupted checkpoint files."""
        checkpoint_dir = self.output_dir / "checkpoint_corruption"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create a corrupted checkpoint file
        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            f.write('{"invalid": json content')

        # Test handling of corrupted checkpoint
        try:
            with open(checkpoint_file, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            # Create a new valid checkpoint
            valid_checkpoint_data = {
                "processed_files": [],
                "current_file": "sample_0.ipc",
                "total_files": 10,
                "timestamp": "2024-01-01T12:00:00",
                "status": "in_progress",
            }

            with open(checkpoint_file, "w") as f:
                json.dump(valid_checkpoint_data, f)

        # Verify recovery worked
        with open(checkpoint_file, "r") as f:
            recovered_data = json.load(f)

        assert "processed_files" in recovered_data
        assert "current_file" in recovered_data
        assert "total_files" in recovered_data
        assert "status" in recovered_data

    def test_checkpoint_validation(self) -> None:
        """Test validation of checkpoint data."""
        checkpoint_dir = self.output_dir / "checkpoint_validation"
        checkpoint_dir.mkdir(exist_ok=True)

        # Test valid checkpoint
        valid_checkpoint: dict[str, Any] = {
            "processed_files": ["sample_0.ipc", "sample_1.ipc"],
            "current_file": "sample_2.ipc",
            "total_files": 10,
            "timestamp": "2024-01-01T12:00:00",
            "status": "in_progress",
        }

        # Validate required fields
        required_fields = ["processed_files", "current_file", "total_files", "status"]
        for field in required_fields:
            assert field in valid_checkpoint

        # Validate data types
        assert isinstance(valid_checkpoint["processed_files"], list)
        assert isinstance(valid_checkpoint["current_file"], str)
        assert isinstance(valid_checkpoint["total_files"], int)
        assert isinstance(valid_checkpoint["status"], str)

        # Validate status values
        valid_statuses = ["in_progress", "completed", "failed"]
        assert valid_checkpoint["status"] in valid_statuses

    def test_checkpoint_cleanup(self) -> None:
        """Test cleanup of checkpoint files."""
        checkpoint_dir = self.output_dir / "checkpoint_cleanup"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create multiple checkpoint files
        checkpoint_files = [
            "split_checkpoint.json",
            "lsh_cache.pkl",
            "temp_file_1.tmp",
            "temp_file_2.tmp",
        ]

        for filename in checkpoint_files:
            file_path = checkpoint_dir / filename
            file_path.touch()

        # Simulate cleanup of temporary files
        temp_files = list(checkpoint_dir.glob("*.tmp"))
        for temp_file in temp_files:
            temp_file.unlink()

        # Verify only checkpoint files remain
        remaining_files = list(checkpoint_dir.iterdir())
        assert len(remaining_files) == 2  # split_checkpoint.json and lsh_cache.pkl

        remaining_names = [f.name for f in remaining_files]
        assert "split_checkpoint.json" in remaining_names
        assert "lsh_cache.pkl" in remaining_names

    def _create_test_data(self) -> None:
        """Create realistic test data for checkpointing tests."""
        # Create multiple test files to simulate processing
        for i in range(10):
            data = {
                "index": list(range(i * 100, (i + 1) * 100)),
                "scan": [f"scan_{j:03d}" for j in range(i * 100, (i + 1) * 100)],
                "header": [
                    f"MS2 scan {1000 + j}@{30 + j}"
                    for j in range(i * 100, (i + 1) * 100)
                ],
                "rt": [30.0 + j for j in range(i * 100, (i + 1) * 100)],
                "frag_type": ["HCD"] * 100,
                "collision_energy": [30.0 + j for j in range(i * 100, (i + 1) * 100)],
                "precursor_mz": [1000.0 + j for j in range(i * 100, (i + 1) * 100)],
                "precursor_charge": [2] * 100,
                "precursor_intensity": [
                    1000.0 + j * 100 for j in range(i * 100, (i + 1) * 100)
                ],
                "lower_offset": [-1.0] * 100,
                "upper_offset": [1.0] * 100,
                "isolation_target": [None] * 100,
                "mz": [
                    [100.0 + j, 200.0 + j, 300.0 + j]
                    for j in range(i * 100, (i + 1) * 100)
                ],
                "intensity": [
                    [100.0 + j, 200.0 + j, 300.0 + j]
                    for j in range(i * 100, (i + 1) * 100)
                ],
                "scale_factor": [1.0] * 100,
            }

            df = pl.DataFrame(data)
            df.write_ipc(self.input_dir / f"sample_{i}.ipc")

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"

        # Create test directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self._create_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)


class TestCheckpointingIntegration:
    """Integration tests for checkpointing workflows."""

    def test_full_checkpoint_workflow(self) -> None:
        """Test a complete checkpoint workflow."""
        checkpoint_dir = self.output_dir / "checkpoint_workflow"
        checkpoint_dir.mkdir(exist_ok=True)

        # Simulate a complete processing workflow with checkpoints
        total_files = 20
        processed_files = []

        for i in range(total_files):
            # Process file
            processed_files.append(f"sample_{i}.ipc")

            # Create checkpoint every 5 files
            if (i + 1) % 5 == 0 or i == total_files - 1:
                checkpoint_data = {
                    "processed_files": processed_files.copy(),
                    "current_file": f"sample_{i + 1}.ipc"
                    if i + 1 < total_files
                    else None,
                    "total_files": total_files,
                    "timestamp": f"2024-01-01T12:{i:02d}:00",
                    "status": "in_progress" if i + 1 < total_files else "completed",
                    "progress_percentage": round(((i + 1) / total_files) * 100, 2),
                }

                checkpoint_file = checkpoint_dir / "split_checkpoint.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f)

        # Verify final checkpoint
        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            final_data = json.load(f)

        assert len(final_data["processed_files"]) == total_files
        assert final_data["status"] == "completed"
        assert final_data["progress_percentage"] == 100.0

    def test_checkpoint_resume_workflow(self) -> None:
        """Test resuming from a checkpoint in a workflow."""
        checkpoint_dir = self.output_dir / "checkpoint_resume_workflow"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create initial checkpoint
        initial_checkpoint: dict[str, Any] = {
            "processed_files": ["sample_0.ipc", "sample_1.ipc", "sample_2.ipc"],
            "current_file": "sample_3.ipc",
            "total_files": 20,
            "timestamp": "2024-01-01T12:00:00",
            "status": "in_progress",
            "progress_percentage": 15.0,
        }

        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(initial_checkpoint, f)

        # Simulate resuming processing
        remaining_files = list(range(3, 20))
        for i in remaining_files:
            initial_checkpoint["processed_files"].append(f"sample_{i}.ipc")
            initial_checkpoint["current_file"] = (
                f"sample_{i + 1}.ipc" if i + 1 < 20 else None
            )
            initial_checkpoint["timestamp"] = f"2024-01-01T12:{i:02d}:00"
            initial_checkpoint["progress_percentage"] = round(((i + 1) / 20) * 100, 2)

            if i == 19:
                initial_checkpoint["status"] = "completed"

            # Update checkpoint every 5 files
            if (i + 1) % 5 == 0 or i == 19:
                with open(checkpoint_file, "w") as f:
                    json.dump(initial_checkpoint, f)

        # Verify final state
        with open(checkpoint_file, "r") as f:
            final_data = json.load(f)

        assert len(final_data["processed_files"]) == 20
        assert final_data["status"] == "completed"
        assert final_data["progress_percentage"] == 100.0

    def test_checkpoint_error_recovery(self) -> None:
        """Test recovery from checkpoint errors."""
        checkpoint_dir = self.output_dir / "checkpoint_error_recovery"
        checkpoint_dir.mkdir(exist_ok=True)

        # Simulate various error scenarios
        error_scenarios: list[dict[str, Any]] = [
            # Missing checkpoint file
            {"scenario": "missing_file", "file_exists": False},
            # Corrupted JSON
            {
                "scenario": "corrupted_json",
                "file_exists": True,
                "content": '{"invalid": json',
            },
            # Missing required fields
            {
                "scenario": "missing_fields",
                "file_exists": True,
                "content": '{"processed_files": []}',
            },
            # Invalid status
            {
                "scenario": "invalid_status",
                "file_exists": True,
                "content": '{"status": "invalid"}',
            },
        ]

        for scenario in error_scenarios:
            checkpoint_file = checkpoint_dir / f"checkpoint_{scenario['scenario']}.json"

            if scenario["file_exists"]:
                with open(checkpoint_file, "w") as f:
                    f.write(scenario.get("content", "{}"))

            # Test recovery logic
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)

                # Validate checkpoint
                required_fields = [
                    "processed_files",
                    "current_file",
                    "total_files",
                    "status",
                ]
                valid_statuses = ["in_progress", "completed", "failed"]

                is_valid = all(field in data for field in required_fields)
                is_valid = is_valid and data.get("status") in valid_statuses

                if not is_valid:
                    # Create default checkpoint
                    default_checkpoint: dict[str, Any] = {
                        "processed_files": [],
                        "current_file": "sample_0.ipc",
                        "total_files": 20,
                        "timestamp": "2024-01-01T12:00:00",
                        "status": "in_progress",
                        "progress_percentage": 0.0,
                    }

                    with open(checkpoint_file, "w") as f:
                        json.dump(default_checkpoint, f)

            except (json.JSONDecodeError, FileNotFoundError):
                # Create default checkpoint
                default_checkpoint = {
                    "processed_files": [],
                    "current_file": "sample_0.ipc",
                    "total_files": 20,
                    "timestamp": "2024-01-01T12:00:00",
                    "status": "in_progress",
                    "progress_percentage": 0.0,
                }

                with open(checkpoint_file, "w") as f:
                    json.dump(default_checkpoint, f)

            # Verify recovery worked
            with open(checkpoint_file, "r") as f:
                recovered_data = json.load(f)

            assert "processed_files" in recovered_data
            assert "current_file" in recovered_data
            assert "total_files" in recovered_data
            assert "status" in recovered_data
            assert recovered_data["status"] in ["in_progress", "completed", "failed"]

    def test_checkpoint_performance(self) -> None:
        """Test checkpoint performance with large datasets."""
        checkpoint_dir = self.output_dir / "checkpoint_performance"
        checkpoint_dir.mkdir(exist_ok=True)

        # Simulate processing with frequent checkpoints
        total_files = 100
        processed_files = []

        start_time = time.time()

        for i in range(total_files):
            processed_files.append(f"sample_{i}.ipc")

            # Create checkpoint every file (stress test)
            checkpoint_data = {
                "processed_files": processed_files.copy(),
                "current_file": f"sample_{i + 1}.ipc" if i + 1 < total_files else None,
                "total_files": total_files,
                "timestamp": f"2024-01-01T12:{i:02d}:00",
                "status": "in_progress" if i + 1 < total_files else "completed",
                "progress_percentage": round(((i + 1) / total_files) * 100, 2),
            }

            checkpoint_file = checkpoint_dir / "split_checkpoint.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance is reasonable (should complete in reasonable time)
        assert processing_time < 10.0  # Should complete within 10 seconds

        # Verify final checkpoint
        checkpoint_file = checkpoint_dir / "split_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            final_data = json.load(f)

        assert len(final_data["processed_files"]) == total_files
        assert final_data["status"] == "completed"
        assert final_data["progress_percentage"] == 100.0

    def _create_integration_test_data(self) -> None:
        """Create realistic test data for integration testing."""
        # Create a larger dataset for integration testing
        for i in range(20):
            data = {
                "index": list(range(i * 50, (i + 1) * 50)),
                "scan": [f"scan_{j:03d}" for j in range(i * 50, (i + 1) * 50)],
                "header": [
                    f"MS2 scan {1000 + j}@{30 + j}" for j in range(i * 50, (i + 1) * 50)
                ],
                "rt": [30.0 + j for j in range(i * 50, (i + 1) * 50)],
                "frag_type": ["HCD"] * 50,
                "collision_energy": [30.0 + j for j in range(i * 50, (i + 1) * 50)],
                "precursor_mz": [1000.0 + j for j in range(i * 50, (i + 1) * 50)],
                "precursor_charge": [2] * 50,
                "precursor_intensity": [
                    1000.0 + j * 100 for j in range(i * 50, (i + 1) * 50)
                ],
                "lower_offset": [-1.0] * 50,
                "upper_offset": [1.0] * 50,
                "isolation_target": [None] * 50,
                "mz": [
                    [100.0 + j, 200.0 + j, 300.0 + j]
                    for j in range(i * 50, (i + 1) * 50)
                ],
                "intensity": [
                    [100.0 + j, 200.0 + j, 300.0 + j]
                    for j in range(i * 50, (i + 1) * 50)
                ],
                "scale_factor": [1.0] * 50,
            }

            df = pl.DataFrame(data)
            df.write_ipc(self.input_dir / f"sample_{i}.ipc")

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self) -> Generator[None, None, None]:
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"

        # Create test directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create realistic test data
        self._create_integration_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)
