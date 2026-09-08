import fnmatch
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class FinetuneScheduler:
    """Scheduler for unfreezing parameters of a model.

    Args:
        model_state_dict (dict): The state dictionary of the model.
        config (DictConfig): The configuration for the scheduler.
        steps_per_epoch (int | None): The number of steps per epoch.
    """

    def __init__(self, model_state_dict: dict, config: DictConfig, steps_per_epoch: int | None = None):
        self.model_state_dict = model_state_dict
        self.config = config
        self.steps_per_epoch = steps_per_epoch

        self.is_verbose = self.config.get("verbose", False)

        self.schedule = self._get_schedule()

        if self.is_verbose:
            logger.info(f"Unfreezing schedule setup with {len(self.schedule)} phases.")
            for i, phase in enumerate(self.schedule):
                logger.info(f" - Phase {i + 1}, global_step {phase['global_step']:,d}, params {phase['params']}")

        self._freeze_parameters()
        self.next_phase: dict[str, Any] | None = self.schedule.pop(0)
        self.step(0)  # Trigger first unfreeze

    def _get_schedule(self) -> list[dict]:
        unfreeze_format = self.config.get("unfreeze_format", "start_epoch")

        phases = self.config.get("unfreeze_schedule", [])

        if len(phases) == 0:
            raise ValueError("No unfreeze_schedule phases specified")

        if any(phase.get(unfreeze_format, None) is None for phase in phases):
            raise ValueError(f"{unfreeze_format} must be specified for each phase")

        schedule = []

        next_global_step = 0
        for phase in phases:
            global_step = 0
            match unfreeze_format:
                case "duration_epochs":
                    if self.steps_per_epoch is None:
                        raise ValueError("steps_per_epoch must be specified for epoch-based scheduling")
                    global_step = next_global_step
                    next_global_step += phase["duration_epochs"] * self.steps_per_epoch
                case "duration_steps":
                    global_step = next_global_step
                    next_global_step += phase["duration_steps"]
                case "start_epoch":
                    if self.steps_per_epoch is None:
                        raise ValueError("steps_per_epoch must be specified for epoch-based scheduling")
                    global_step = phase["start_epoch"] * self.steps_per_epoch
                case "start_step":
                    global_step = phase["start_step"]
                case _:
                    raise ValueError(f"Invalid unfreeze format: {unfreeze_format}")

            schedule.append({"global_step": global_step, "completed": False, "params": phase["params"]})

        # Check schedule is valid
        if schedule[0]["global_step"] != 0:
            raise ValueError("First phase must start at global step 0")

        last_step = schedule[0]["global_step"]
        for phase in schedule:
            step = phase["global_step"]
            if step < last_step:
                raise ValueError("Phases must be in increasing order of steps/epochs")
            last_step = step

        return schedule

    def _freeze_parameters(self) -> None:
        logger.info("Freezing model parameters")
        num_params = 0
        num_layers = 0
        for _, param in self.model_state_dict.items():
            param.requires_grad = False
            num_params += param.numel()
            num_layers += 1
        logger.info(f"Frozen {num_params:,d} parameters in {num_layers:,d} layers")

    def _unfreeze(self, param_patterns: list[str]) -> None:
        logger.info(f"Unfreezing parameters: {param_patterns}")
        num_params = 0
        num_layers = 0
        for name, param in self.model_state_dict.items():
            for pattern in param_patterns:
                if pattern == "*" or fnmatch.fnmatch(name, pattern):
                    param.requires_grad = True
                    num_params += param.numel()
                    num_layers += 1
                    break
        logger.info(f"Unfrozen {num_params:,d} parameters in {num_layers:,d} layers")

    def step(self, global_step: int) -> None:
        """Step the unfreezing scheduler.

        Args:
            global_step (int): The global step of the model.
        """
        if self.next_phase is None or global_step < self.next_phase["global_step"]:
            return
        self._unfreeze(self.next_phase["params"])
        if len(self.schedule) > 0:
            self.next_phase = self.schedule.pop(0)
        else:
            self.next_phase = None


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : int
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        # Cosine annealing after a constant period
        decay = self.warmup / self.max_iters
        if epoch <= self.warmup and self.warmup > 0:
            lr_factor = 1 * (epoch / self.warmup)
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * ((epoch - (decay * self.max_iters)) / ((1 - decay) * self.max_iters))))

        return lr_factor


class CosineWarmupHoldScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup -> optional hold at max LR -> cosine decay to floor.

    Accepts either fractional or absolute phase lengths.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    max_iters : int
        The total number of iterations.
    warmup_steps : int
        The number of warmup steps (default: 0).
    hold_steps : int
        The number of steps to hold at max LR (default: 0).
    min_lr_factor : float
        The minimum LR factor (floor) relative to base LR (default: 0.0).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_iters: int,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        min_lr_factor: float = 0.0,
    ) -> None:
        self.max_iters = int(max_iters)
        self.warmup_steps = int(max(0, warmup_steps))
        self.hold_steps = int(max(0, hold_steps))
        self.min_lr_factor = float(min_lr_factor)
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        f = self._lr_factor(self.last_epoch)
        return [base_lr * f for base_lr in self.base_lrs]

    def _lr_factor(self, step: int) -> float:
        """Calculate the LR factor at the current step."""
        T = max(1, self.max_iters)  # noqa: N806
        s = min(max(step, 0), T)

        # Phase boundaries
        w = self.warmup_steps
        h = self.hold_steps
        decay_len = max(1, T - w - h)

        if s < w:  # Warmup
            return s / max(1, w)
        elif s < w + h:  # Hold
            return 1.0
        else:  # Cosine decay to floor
            t = (s - w - h) / decay_len  # [0, 1]
            return self.min_lr_factor + (1 - self.min_lr_factor) * 0.5 * (1 + np.cos(np.pi * t))  # type: ignore[no-any-return]
