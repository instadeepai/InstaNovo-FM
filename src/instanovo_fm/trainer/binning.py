"""M/Z Binning Strategies for Foundational Model Classification Head.

This module provides multiple binning strategies for the hierarchical m/z classification task:
- fixed_da: Fixed-width bins in Daltons (existing behavior)
- fixed_ppm: Approximately constant PPM resolution
- adaptive: Smooth parametric bin width function (physics-motivated)

All strategies support hierarchical (group, offset) classification and are
checkpoint-safe and device-safe.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class BinningStrategy(ABC):
    """Base class for m/z binning strategies.

    All strategies must support:
    - Compute monotonic bin edges covering [min_mz, max_mz]
    - Map m/z values to bin indices (forward)
    - Map bin indices to m/z centers (inverse)
    - Hierarchical decomposition into (group, offset) pairs
    - Serialization for checkpoints
    """

    def __init__(self, min_mz: float, max_mz: float, bin_group_size: int) -> None:
        """Initialize binning strategy.

        Args:
            min_mz: Minimum m/z value in Daltons
            max_mz: Maximum m/z value in Daltons
            bin_group_size: Number of bins per group (for hierarchical classification)
        """
        if min_mz >= max_mz:
            raise ValueError(f"min_mz ({min_mz}) must be < max_mz ({max_mz})")
        if bin_group_size <= 0:
            raise ValueError(f"bin_group_size ({bin_group_size}) must be > 0")

        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_group_size = bin_group_size
        self._bin_edges = None

    @abstractmethod
    def compute_bin_edges(self) -> torch.Tensor:
        """Compute bin edges covering [min_mz, max_mz].

        Returns:
            Tensor of shape [n_bins+1] with monotonically increasing edges
        """
        pass

    @property
    def bin_edges(self) -> torch.Tensor:
        """Get or compute bin edges."""
        if self._bin_edges is None:
            self._bin_edges = self.compute_bin_edges()
        return self._bin_edges

    @property
    def n_bins(self) -> int:
        """Total number of bins."""
        return len(self.bin_edges) - 1

    @property
    def n_groups(self) -> int:
        """Number of bin groups (for hierarchical classification)."""
        return (self.n_bins + self.bin_group_size - 1) // self.bin_group_size

    @property
    def last_group_size(self) -> int:
        """Number of bins in the last group."""
        rem = self.n_bins % self.bin_group_size
        return rem if rem > 0 else self.bin_group_size

    def mz_to_bin(self, mz_values: torch.Tensor) -> torch.Tensor:
        """Map m/z values to bin indices.

        For mz in [edges[i], edges[i+1]), assign bin index i.

        Boundary behavior:
        - mz == edges[k] → bin k (value at left edge goes to that bin)
        - mz == edges[-1] (max_mz) → bin n_bins-1 (clamped to last bin)
        - mz < edges[0] → bin 0 (clamped)
        - mz > edges[-1] → bin n_bins-1 (clamped)

        Args:
            mz_values: Tensor of m/z values in Daltons

        Returns:
            Tensor of bin indices [0, n_bins-1]
        """
        edges = self.bin_edges.to(mz_values.device)
        # Use right=True so mz == edges[k] returns k+1, then -1 gives bin k
        # This ensures values at left edge of bin k go to bin k (not k-1)
        bin_indices = torch.searchsorted(edges, mz_values, right=True) - 1
        return torch.clamp(bin_indices, 0, self.n_bins - 1)

    def bin_to_mz(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """Map bin indices to m/z centers.

        Args:
            bin_indices: Tensor of bin indices

        Returns:
            Tensor of m/z centers in Daltons
        """
        edges = self.bin_edges.to(bin_indices.device)
        # Bin center = (left_edge + right_edge) / 2
        left = edges[bin_indices]
        right = edges[bin_indices + 1]
        return (left + right) / 2.0

    def mz_to_bin_groups(self, mz_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map m/z values to hierarchical (group, offset) indices.

        Args:
            mz_values: Tensor of m/z values in Daltons

        Returns:
            Tuple of (group_indices, offset_indices)
        """
        bin_indices = self.mz_to_bin(mz_values)
        group_indices = bin_indices // self.bin_group_size
        offset_indices = bin_indices % self.bin_group_size
        return group_indices, offset_indices

    def bin_groups_to_mz(self, group_indices: torch.Tensor, offset_indices: torch.Tensor) -> torch.Tensor:
        """Map hierarchical (group, offset) indices to m/z centers.

        Args:
            group_indices: Tensor of group indices
            offset_indices: Tensor of offset indices within groups

        Returns:
            Tensor of m/z centers in Daltons
        """
        bin_indices = group_indices * self.bin_group_size + offset_indices
        return self.bin_to_mz(bin_indices)

    def state_dict(self) -> Dict[str, Any]:
        """Save binning configuration for checkpoints."""
        return {
            "strategy": self.__class__.__name__,
            "min_mz": self.min_mz,
            "max_mz": self.max_mz,
            "bin_group_size": self.bin_group_size,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_mz={self.min_mz}, max_mz={self.max_mz}, n_bins={self.n_bins}, n_groups={self.n_groups})"


class FixedDaBinning(BinningStrategy):
    """Fixed-width binning in Daltons (existing behavior).

    This is the default binning strategy for backward compatibility.
    All bins have constant width in Daltons, which means PPM resolution
    varies across the m/z range (better at low m/z, worse at high m/z).

    Example:
        bin_size=0.02 Da → constant 0.02 Da bins
        At m/z=100: ~200 PPM resolution
        At m/z=2000: ~10 PPM resolution
    """

    def __init__(self, min_mz: float, max_mz: float, bin_size: float, bin_group_size: int) -> None:
        """Initialize fixed Da binning.

        Args:
            min_mz: Minimum m/z value in Daltons
            max_mz: Maximum m/z value in Daltons
            bin_size: Width of each bin in Daltons
            bin_group_size: Number of bins per group
        """
        super().__init__(min_mz, max_mz, bin_group_size)
        if bin_size <= 0:
            raise ValueError(f"bin_size ({bin_size}) must be > 0")
        self.bin_size = bin_size

    def compute_bin_edges(self) -> torch.Tensor:
        """Compute uniform bin edges."""
        n_bins = int((self.max_mz - self.min_mz) / self.bin_size)
        # Use linspace for exact min/max coverage
        edges = torch.linspace(self.min_mz, self.max_mz, n_bins + 1)
        return edges

    def mz_to_bin(self, mz_values: torch.Tensor) -> torch.Tensor:
        """Fast path: use formula instead of binary search."""
        bin_indices = torch.floor((mz_values - self.min_mz) / self.bin_size).long()
        return torch.clamp(bin_indices, 0, self.n_bins - 1)

    def bin_to_mz(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """Fast path: use formula."""
        return self.min_mz + (bin_indices.float() + 0.5) * self.bin_size

    def state_dict(self) -> Dict[str, Any]:
        """Save configuration for checkpoints."""
        state = super().state_dict()
        state["bin_size"] = self.bin_size
        return state


class FixedPpmBinning(BinningStrategy):
    """Constant PPM resolution binning.

    Bins are constructed to have approximately constant PPM width across the m/z range.
    This provides uniform relative resolution, which is often desirable for mass spectrometry.

    Algorithm: Iteratively construct edges where each bin width = current_mz * ppm_target / 1e6

    Example:
        ppm_target=10 PPM
        At m/z=100: bin width ≈ 0.001 Da
        At m/z=2000: bin width ≈ 0.020 Da
    """

    def __init__(self, min_mz: float, max_mz: float, ppm_target: float, bin_group_size: int) -> None:
        """Initialize fixed PPM binning.

        Args:
            min_mz: Minimum m/z value in Daltons
            max_mz: Maximum m/z value in Daltons
            ppm_target: Target PPM resolution (e.g., 10.0 for ±10 PPM bins)
            bin_group_size: Number of bins per group
        """
        super().__init__(min_mz, max_mz, bin_group_size)
        if ppm_target <= 0:
            raise ValueError(f"ppm_target ({ppm_target}) must be > 0")
        self.ppm_target = ppm_target

    def compute_bin_edges(self) -> torch.Tensor:
        """Iteratively construct edges with constant PPM spacing."""
        edges = [self.min_mz]
        current = self.min_mz

        while current < self.max_mz:
            # Bin width = current_mz * ppm_target / 1e6
            width = current * self.ppm_target / 1e6
            width = max(width, 1e-6)  # Safety: prevent zero width

            next_edge = current + width

            # Stop if we've exceeded max_mz
            if next_edge > self.max_mz:
                break

            edges.append(next_edge)
            current = next_edge

        # Ensure last edge is exactly max_mz
        edges.append(self.max_mz)

        return torch.tensor(edges, dtype=torch.float32)

    def state_dict(self) -> Dict[str, Any]:
        """Save configuration for checkpoints."""
        state = super().state_dict()
        state["ppm_target"] = self.ppm_target
        return state


class AdaptiveBinning(BinningStrategy):
    """Smooth m/z-dependent binning with parametric bin width function.

    Bin width follows a physics-motivated model that transitions smoothly
    between constant-Da at low m/z and constant-PPM at high m/z. The bin
    width function is C-infinity smooth with no discontinuities or knots.

    Mass measurement error has two independent components combining in quadrature:
    - Additive (constant Da): electronic noise, ADC quantization
    - Multiplicative (constant PPM): centroiding uncertainty, calibration

    Supported functions:
    - "hyperbolic": w(mz) = sqrt(da_floor^2 + (mz * ppm_asymptote/1e6)^2)
      Physics-motivated from quadrature of Da + PPM error components.
    - "power_law": w(mz) = scale * mz^exponent
      Flexible interpolation between Da (exp=0) and PPM (exp=1).
    - "linear": w(mz) = da_floor + mz * ppm_slope/1e6
      Additive Da + PPM model (linear combination instead of quadrature).

    Example (hyperbolic defaults for mixed CID+HCD data):
        da_floor=0.01, ppm_asymptote=10.0
        At m/z=100: width ≈ 0.01 Da (Da-dominated)
        At m/z=2000: width ≈ 0.02 Da (PPM-dominated)
        Da bounds: [0.005, 0.12]
    """

    SUPPORTED_FUNCTIONS = ("hyperbolic", "power_law", "linear")

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        function: str = "hyperbolic",
        da_floor: float = 0.01,
        ppm_asymptote: float = 10.0,
        ppm_slope: float = 10.0,
        scale: float = 0.001,
        exponent: float = 0.5,
        min_da: float = 0.005,
        max_da: float = 0.12,
        bin_group_size: int = 100,
    ) -> None:
        """Initialize adaptive binning.

        Args:
            min_mz: Minimum m/z value in Daltons
            max_mz: Maximum m/z value in Daltons
            function: Bin width function type ("hyperbolic", "power_law", "linear")
            da_floor: Constant Da component for hyperbolic/linear (dominates at low m/z)
            ppm_asymptote: PPM asymptote for hyperbolic (dominates at high m/z)
            ppm_slope: PPM slope for linear function
            scale: Scale factor for power_law function
            exponent: Exponent for power_law function (0=Da-like, 1=PPM-like)
            min_da: Minimum bin width safety floor (Da)
            max_da: Maximum bin width safety ceiling (Da)
            bin_group_size: Number of bins per group
        """
        super().__init__(min_mz, max_mz, bin_group_size)

        if function not in self.SUPPORTED_FUNCTIONS:
            raise ValueError(f"Unknown function '{function}'. Supported: {', '.join(self.SUPPORTED_FUNCTIONS)}")

        if min_da <= 0:
            raise ValueError(f"min_da ({min_da}) must be > 0")
        if max_da <= min_da:
            raise ValueError(f"max_da ({max_da}) must be > min_da ({min_da})")

        self.function = function
        self.da_floor = da_floor
        self.ppm_asymptote = ppm_asymptote
        self.ppm_slope = ppm_slope
        self.scale = scale
        self.exponent = exponent
        self.min_da = min_da
        self.max_da = max_da

        # Validate function-specific parameters
        if function == "hyperbolic":
            if da_floor <= 0:
                raise ValueError(f"da_floor ({da_floor}) must be > 0")
            if ppm_asymptote <= 0:
                raise ValueError(f"ppm_asymptote ({ppm_asymptote}) must be > 0")
        elif function == "power_law":
            if scale <= 0:
                raise ValueError(f"scale ({scale}) must be > 0")
            if exponent < 0:
                raise ValueError(f"exponent ({exponent}) must be >= 0")
        elif function == "linear":
            if da_floor <= 0:
                raise ValueError(f"da_floor ({da_floor}) must be > 0")
            if ppm_slope <= 0:
                raise ValueError(f"ppm_slope ({ppm_slope}) must be > 0")

    def _get_bin_width_da(self, mz: float) -> float:
        """Compute bin width in Da at given m/z using the parametric function.

        Args:
            mz: m/z value in Daltons

        Returns:
            Bin width in Da (before safety clamping)
        """
        if self.function == "hyperbolic":
            return math.sqrt(self.da_floor**2 + (mz * self.ppm_asymptote / 1e6) ** 2)
        elif self.function == "power_law":
            return float(self.scale * mz**self.exponent)
        elif self.function == "linear":
            return self.da_floor + mz * self.ppm_slope / 1e6
        else:
            raise ValueError(f"Unknown function: {self.function}")

    def compute_bin_edges(self) -> torch.Tensor:
        """Construct edges with parametric spacing and Da safety bounds.

        Uses float64 internally for numerical precision, returns float32.

        Returns:
            Tensor of bin edges covering [min_mz, max_mz]

        Raises:
            RuntimeError: If bin count exceeds safety limit (indicates infinite loop)
        """
        max_bins = int((self.max_mz - self.min_mz) / self.min_da) + 1000

        edges: list[float] = []
        current: float = float(self.min_mz)

        while current < self.max_mz:
            edges.append(current)

            if len(edges) > max_bins:
                raise RuntimeError(f"Exceeded maximum bin count ({max_bins}). Check parameters: function={self.function}, min_da={self.min_da}")

            width = self._get_bin_width_da(current)
            # Apply Da safety bounds; use 1.0001 epsilon for strict monotonicity
            width = max(min(width, self.max_da), self.min_da * 1.0001)

            next_edge = current + width

            if next_edge >= self.max_mz:
                remaining = self.max_mz - current
                if remaining >= self.min_da * 0.999:
                    edges.append(float(self.max_mz))
                else:
                    if len(edges) > 0:
                        edges[-1] = float(self.max_mz)
                    else:
                        edges.append(float(self.max_mz))
                break

            current = next_edge

        if edges[-1] != self.max_mz:
            edges.append(float(self.max_mz))

        return torch.tensor(edges, dtype=torch.float32)

    def state_dict(self) -> Dict[str, Any]:
        """Save configuration for checkpoints."""
        state = super().state_dict()
        state["function"] = self.function
        state["min_da"] = self.min_da
        state["max_da"] = self.max_da

        if self.function == "hyperbolic":
            state["da_floor"] = self.da_floor
            state["ppm_asymptote"] = self.ppm_asymptote
        elif self.function == "power_law":
            state["scale"] = self.scale
            state["exponent"] = self.exponent
        elif self.function == "linear":
            state["da_floor"] = self.da_floor
            state["ppm_slope"] = self.ppm_slope

        return state

    def __repr__(self) -> str:
        if self.function == "hyperbolic":
            params = f"da_floor={self.da_floor}, ppm_asymptote={self.ppm_asymptote}"
        elif self.function == "power_law":
            params = f"scale={self.scale}, exponent={self.exponent}"
        elif self.function == "linear":
            params = f"da_floor={self.da_floor}, ppm_slope={self.ppm_slope}"
        else:
            params = ""

        return (
            f"AdaptiveBinning(function={self.function}, {params}, "
            f"da_range=[{self.min_da}, {self.max_da}], "
            f"mz_range=[{self.min_mz}, {self.max_mz}], "
            f"n_bins={self.n_bins}, n_groups={self.n_groups})"
        )


def create_binning_strategy(config: Dict[str, Any], min_mz: float, max_mz: float) -> BinningStrategy:
    """Factory for creating binning strategies from config.

    Expects nested config format:
        mz_head:
          binning:
            strategy: "fixed_da"
            bin_size: 0.02
          bin_group_size: 50

    Note: Old checkpoint configs are migrated to this format in
    FoundationModel.load() before reaching this function.

    Args:
        config: Full model config dictionary
        min_mz: Minimum m/z value in Daltons
        max_mz: Maximum m/z value in Daltons

    Returns:
        Initialized binning strategy

    Raises:
        ValueError: If strategy is unknown, required parameters are missing,
            or the config is missing the mz_head.binning key.
    """
    mz_head_cfg = config.get("mz_head", {})
    binning_cfg = mz_head_cfg.get("binning", {})

    if not binning_cfg:
        raise ValueError("Missing mz_head.binning config. If loading an old checkpoint, ensure FoundationModel.load() migrates the config first.")

    strategy = binning_cfg.get("strategy", "fixed_da")
    bin_group_size = mz_head_cfg.get("bin_group_size", 50)

    if strategy == "fixed_da":
        bin_size = binning_cfg.get("bin_size")
        if bin_size is None:
            raise ValueError("fixed_da strategy requires 'bin_size' parameter")

        logger.info(f"Creating FixedDaBinning: bin_size={bin_size} Da, range=[{min_mz}, {max_mz}] Da")
        return FixedDaBinning(min_mz, max_mz, bin_size, bin_group_size)

    elif strategy == "fixed_ppm":
        ppm_target = binning_cfg.get("ppm_target")
        if ppm_target is None:
            raise ValueError("fixed_ppm strategy requires 'ppm_target' parameter")

        logger.info(f"Creating FixedPpmBinning: ppm_target={ppm_target} PPM, range=[{min_mz}, {max_mz}] Da")
        return FixedPpmBinning(min_mz, max_mz, ppm_target, bin_group_size)

    elif strategy == "adaptive":
        function = binning_cfg.get("function", "hyperbolic")

        logger.debug(
            f"AdaptiveBinning: function={function}, "
            f"da_range=[{binning_cfg.get('min_da', 0.005)}, {binning_cfg.get('max_da', 0.12)}], "
            f"range=[{min_mz}, {max_mz}] Da"
        )

        return AdaptiveBinning(
            min_mz,
            max_mz,
            function=function,
            da_floor=binning_cfg.get("da_floor", 0.01),
            ppm_asymptote=binning_cfg.get("ppm_asymptote", 10.0),
            ppm_slope=binning_cfg.get("ppm_slope", 10.0),
            scale=binning_cfg.get("scale", 0.001),
            exponent=binning_cfg.get("exponent", 0.5),
            min_da=binning_cfg.get("min_da", 0.005),
            max_da=binning_cfg.get("max_da", 0.12),
            bin_group_size=bin_group_size,
        )

    elif strategy == "log_hybrid":
        raise ValueError(
            "The 'log_hybrid' strategy has been replaced by 'adaptive'. "
            "Please update your config to use:\n\n"
            "  binning:\n"
            "    strategy: adaptive\n"
            "    function: hyperbolic\n"
            "    da_floor: 0.01\n"
            "    ppm_asymptote: 10.0\n"
            "    min_da: 0.005\n"
            "    max_da: 0.12\n"
        )

    else:
        raise ValueError(f"Unknown binning strategy: {strategy}. Supported strategies: fixed_da, fixed_ppm, adaptive")
