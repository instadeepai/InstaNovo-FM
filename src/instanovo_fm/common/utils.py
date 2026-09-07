import time
from typing import Any, Dict

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

logger = ColorLog(console, __name__).logger


# Used to record additional training state parameters
# This is only used for accelerate.save_state and
# accelerate.load_state. (resuming runs)
class TrainingState:
    """Training state for tracking training progress.

    This class is used by Accelerate to save and load training state during
    checkpointing and resuming training runs. It tracks the current epoch
    and global step of training.
    """

    def __init__(self) -> None:
        """Initialize training state with zeroed counters."""
        self._global_step: int = 0
        self._epoch: int = 0

    @property
    def global_step(self) -> int:
        """Get the current global step."""
        return self._global_step

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._epoch

    def state_dict(self) -> dict[str, Any]:
        """Get the state dictionary for saving.

        Returns:
            dict[str, Any]: Dictionary containing the current training state.
        """
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dictionary.

        Args:
            state_dict: Dictionary containing the training state to load.
        """
        self._global_step = state_dict["global_step"]
        self._epoch = state_dict["epoch"]

    def step(self) -> None:
        """Step the global step."""
        self._global_step += 1

    def step_epoch(self) -> None:
        """Step the epoch."""
        self._epoch += 1

    def unstep_epoch(self) -> None:
        """Unstep the epoch."""
        self._epoch -= 1


class Timer:
    """Timer for training and validation."""

    def __init__(self, total_steps: int | None = None):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0

    def start(self) -> None:
        """Restart the timer."""
        self.start_time = time.time()
        self.current_step = 0

    def step(self) -> None:
        """Step the timer."""
        self.current_step += 1
        self.last_time = time.time()

    def get_delta(self) -> float:
        """Get the time delta since the timer was started."""
        return self.last_time - self.start_time

    def get_eta(self, current_step: int | None = None) -> float:
        """Get the estimated time to completion."""
        if self.total_steps is None:
            raise ValueError("Total steps is not set.")
        current_step = current_step or self.current_step
        if current_step == 0:
            return 0
        return self.get_delta() / current_step * max(self.total_steps - current_step, 0)

    def get_total_time(self) -> float:
        """Get the total time expected to complete all steps."""
        if self.total_steps is None:
            raise ValueError("Total steps is not set.")
        return self.get_delta() / self.current_step * self.total_steps

    def get_rate(self, current_step: int | None = None) -> float:
        """Get the rate of steps per second."""
        current_step = current_step or self.current_step
        return current_step / self.get_delta()

    def get_step_time(self, current_step: int | None = None) -> float:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return self.get_delta() / current_step

    def get_time_str(self) -> str:
        """Get the time delta since the timer was started."""
        return Timer._format_time(self.get_delta())

    def get_eta_str(self, current_step: int | None = None) -> str:
        """Get the estimated time to completion."""
        current_step = current_step or self.current_step
        return Timer._format_time(self.get_eta(current_step))

    def get_total_time_str(self) -> str:
        """Get the total time expected to complete all steps."""
        return Timer._format_time(self.get_total_time())

    def get_rate_str(self, current_step: int | None = None) -> str:
        """Get the rate of steps per second."""
        current_step = current_step or self.current_step
        return f"{self.get_rate(current_step):.2f} steps/s"

    def get_step_time_rate_str(self, current_step: int | None = None) -> str:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return f"{self.get_step_time(current_step):.2f} s/step"

    def get_step_time_str(self, current_step: int | None = None) -> str:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return Timer._format_time(self.get_step_time(current_step))

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to HH:MM:SS."""
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _get_filepath_mapping(file_groups: Dict[str, str]) -> Dict[str, str]:
    """Get filepath mapping for validation groups."""
    group_mapping = {}
    for group, path in file_groups.items():
        for fp in SpectrumDataFrame._convert_file_paths(path):
            group_mapping[fp] = group
    return group_mapping
