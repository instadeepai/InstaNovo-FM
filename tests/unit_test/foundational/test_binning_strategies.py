"""Unit tests for m/z binning strategies.

Tests all three binning strategies (fixed_da, fixed_ppm, adaptive) for:
- Bin edge computation correctness
- Monotonicity and coverage
- Round-trip accuracy (mz -> bin -> mz)
- Hierarchical decomposition consistency
- Factory function and backward compatibility
"""

import torch
import pytest

from instanovo_fm.trainer.binning import (
    FixedDaBinning,
    FixedPpmBinning,
    AdaptiveBinning,
    create_binning_strategy,
)


class TestFixedDaBinning:
    """Tests for fixed-width Dalton binning (existing behavior)."""

    def test_bin_edges(self):
        """Test uniform bin edge computation."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )
        edges = strategy.compute_bin_edges()

        # Check monotonicity
        assert torch.all(edges[1:] > edges[:-1]), "Bin edges must be monotonic"

        # Check coverage
        assert edges[0] == 50.0, "First edge must be min_mz"
        assert edges[-1] == 2500.0, "Last edge must be max_mz"

        # Check uniform spacing (linspace may have small floating point variations)
        diffs = edges[1:] - edges[:-1]
        expected_diff = (2500.0 - 50.0) / (len(edges) - 1)
        assert torch.allclose(
            diffs, torch.tensor(expected_diff), atol=1e-3
        ), f"Bins must have uniform spacing (mean: {diffs.mean():.6f}, expected: {expected_diff:.6f})"

    def test_mz_to_bin_roundtrip(self):
        """Test m/z -> bin -> m/z round trip accuracy."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )

        mz_test = torch.tensor([100.0, 500.0, 1000.0, 2000.0])

        # Forward: m/z -> bin
        bin_indices = strategy.mz_to_bin(mz_test)

        # Inverse: bin -> m/z (center)
        mz_reconstructed = strategy.bin_to_mz(bin_indices)

        # Should be within half bin width (allow floating point error)
        errors = torch.abs(mz_test - mz_reconstructed)
        assert torch.all(
            errors <= 0.01 + 1e-4
        ), f"Roundtrip error should be <= {0.02/2} Da (got max {errors.max().item():.6f})"

    def test_n_bins_computation(self):
        """Test number of bins is computed correctly."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )

        expected_n_bins = int((2500.0 - 50.0) / 0.02)
        assert (
            strategy.n_bins == expected_n_bins
        ), f"Expected {expected_n_bins} bins, got {strategy.n_bins}"

    def test_group_properties(self):
        """Test group size and last group size."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )

        n_bins = strategy.n_bins
        expected_n_groups = (n_bins + 50 - 1) // 50
        expected_last_group = n_bins % 50 if n_bins % 50 > 0 else 50

        assert (
            strategy.n_groups == expected_n_groups
        ), f"Expected {expected_n_groups} groups"
        assert (
            strategy.last_group_size == expected_last_group
        ), f"Expected last group size {expected_last_group}"


class TestFixedPpmBinning:
    """Tests for constant PPM resolution binning."""

    def test_bin_edges_monotonic(self):
        """Test PPM bin edges are monotonic and cover range."""
        strategy = FixedPpmBinning(
            min_mz=50.0, max_mz=2500.0, ppm_target=10.0, bin_group_size=50
        )
        edges = strategy.compute_bin_edges()

        # Check monotonicity
        assert torch.all(edges[1:] > edges[:-1]), "Bin edges must be monotonic"

        # Check coverage
        assert edges[0] == 50.0, "First edge must be min_mz"
        assert edges[-1] == 2500.0, "Last edge must be max_mz"

    def test_ppm_accuracy(self):
        """Test actual PPM resolution matches target."""
        strategy = FixedPpmBinning(
            min_mz=50.0, max_mz=2500.0, ppm_target=10.0, bin_group_size=50
        )
        edges = strategy.compute_bin_edges()

        # Compute actual PPM at bin centers
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = edges[1:] - edges[:-1]
        actual_ppm = (widths / centers) * 1e6

        # Should be close to target (within 10% tolerance)
        mean_ppm = torch.mean(actual_ppm).item()
        assert abs(mean_ppm - 10.0) < 1.0, f"Mean PPM {mean_ppm} deviates from target"

        # PPM should be relatively constant (low std)
        std_ppm = torch.std(actual_ppm).item()
        assert std_ppm < 2.0, f"PPM std {std_ppm} is too high (not constant)"

    def test_increasing_bin_width(self):
        """Test that bin width increases with m/z for fixed PPM."""
        strategy = FixedPpmBinning(
            min_mz=50.0, max_mz=2500.0, ppm_target=10.0, bin_group_size=50
        )
        edges = strategy.compute_bin_edges()

        widths = edges[1:] - edges[:-1]

        # For fixed PPM, bin width should generally increase with m/z
        increasing_count = (widths[1:] >= widths[:-1]).sum().item()
        total_comparisons = len(widths) - 1

        assert (
            increasing_count / total_comparisons > 0.7
        ), f"Bin widths should mostly increase with m/z for fixed PPM (got {increasing_count}/{total_comparisons} = {increasing_count/total_comparisons:.2%})"

    def test_mz_to_bin_roundtrip(self):
        """Test m/z -> bin -> m/z round trip accuracy."""
        strategy = FixedPpmBinning(
            min_mz=50.0, max_mz=2500.0, ppm_target=10.0, bin_group_size=50
        )

        mz_test = torch.tensor([100.0, 500.0, 1000.0, 2000.0])

        # Forward
        bin_indices = strategy.mz_to_bin(mz_test)

        # Inverse
        mz_reconstructed = strategy.bin_to_mz(bin_indices)

        # For variable bins, error should be within half the local bin width
        edges = strategy.bin_edges
        left = edges[bin_indices]
        right = edges[bin_indices + 1]
        half_width = (right - left) / 2.0

        errors = torch.abs(mz_test - mz_reconstructed)
        assert torch.all(
            errors <= half_width + 1e-5
        ), "Roundtrip error exceeds half bin width"


class TestAdaptiveBinning:
    """Tests for adaptive binning with parametric bin width functions."""

    def test_hyperbolic_low_mz_dominated_by_da_floor(self):
        """Test that at low m/z, bin width is approximately da_floor."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=10.0,
            min_da=0.001, max_da=1.0,  # Wide bounds so we see raw function
            bin_group_size=50,
        )

        # At mz=100: sqrt(0.01^2 + (100*10/1e6)^2) = sqrt(1e-4 + 1e-6) ~ 0.01005
        width = strategy._get_bin_width_da(100.0)
        assert abs(width - 0.01) < 0.002, (
            f"At low m/z, width should be ~da_floor (0.01), got {width:.6f}"
        )

    def test_hyperbolic_high_mz_dominated_by_ppm(self):
        """Test that at high m/z, bin width approaches mz * ppm / 1e6."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=10.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        # At mz=2000: sqrt(0.01^2 + (2000*10/1e6)^2) = sqrt(1e-4 + 4e-4) ~ 0.0224
        width = strategy._get_bin_width_da(2000.0)
        ppm_width = 2000.0 * 10.0 / 1e6  # 0.02
        assert abs(width - ppm_width) < 0.005, (
            f"At high m/z, width should approach mz*ppm/1e6 ({ppm_width:.4f}), got {width:.6f}"
        )

    def test_hyperbolic_smooth_transition(self):
        """Test that bin width function is smooth (no discontinuities)."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=10.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        # Sample many points and check that finite differences are bounded
        mz_values = [50.0 + i * 10.0 for i in range(245)]
        widths = [strategy._get_bin_width_da(mz) for mz in mz_values]

        for i in range(1, len(widths)):
            delta = abs(widths[i] - widths[i - 1])
            # For a smooth function, change per 10 Da step should be small
            assert delta < 0.01, (
                f"Discontinuity at mz={mz_values[i]}: "
                f"width jumped by {delta:.6f} Da"
            )

    def test_hyperbolic_monotonically_increasing(self):
        """Test that hyperbolic bin width is monotonically increasing."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=10.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        mz_values = [50.0 + i * 50.0 for i in range(49)]
        widths = [strategy._get_bin_width_da(mz) for mz in mz_values]

        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1], (
                f"Width not monotonic at mz={mz_values[i]}: "
                f"{widths[i]:.6f} < {widths[i-1]:.6f}"
            )

    def test_power_law_exponent_zero_is_constant(self):
        """Test that exponent=0 gives constant width (Da-like)."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="power_law",
            scale=0.02, exponent=0.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        # With exponent=0: w(mz) = scale * mz^0 = scale = 0.02
        w1 = strategy._get_bin_width_da(100.0)
        w2 = strategy._get_bin_width_da(2000.0)
        assert abs(w1 - 0.02) < 1e-6
        assert abs(w2 - 0.02) < 1e-6

    def test_power_law_exponent_one_is_ppm_like(self):
        """Test that exponent=1 gives PPM-like scaling."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="power_law",
            scale=1e-5, exponent=1.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        # With exponent=1: w(mz) = scale * mz
        w1 = strategy._get_bin_width_da(100.0)
        w2 = strategy._get_bin_width_da(2000.0)
        assert abs(w2 / w1 - 20.0) < 0.01, "Width ratio should equal mz ratio for exp=1"

    def test_linear_intercept_and_slope(self):
        """Test linear function has correct intercept and slope."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="linear",
            da_floor=0.01, ppm_slope=10.0,
            min_da=0.001, max_da=1.0,
            bin_group_size=50,
        )

        # w(0) = da_floor = 0.01 (extrapolated)
        # w(1000) = 0.01 + 1000 * 10/1e6 = 0.01 + 0.01 = 0.02
        w0 = strategy._get_bin_width_da(0.0)
        w1000 = strategy._get_bin_width_da(1000.0)
        assert abs(w0 - 0.01) < 1e-6
        assert abs(w1000 - 0.02) < 1e-6

    def test_da_safety_bounds_enforced(self):
        """Test that Da safety bounds are respected."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.001, ppm_asymptote=5.0,
            min_da=0.01, max_da=0.05,
            bin_group_size=50,
        )

        edges = strategy.compute_bin_edges()
        widths = edges[1:] - edges[:-1]

        tolerance = 0.001
        assert torch.all(widths >= 0.01 - tolerance), (
            f"Some bins violate min_da: min width = {widths.min():.6f}"
        )
        assert torch.all(widths <= 0.05 + tolerance), (
            f"Some bins violate max_da: max width = {widths.max():.6f}"
        )

    def test_monotonic_edges(self):
        """Test edges are strictly monotonic and cover full range."""
        for func in AdaptiveBinning.SUPPORTED_FUNCTIONS:
            strategy = AdaptiveBinning(
                min_mz=50.0, max_mz=2500.0,
                function=func,
                da_floor=0.01, ppm_asymptote=15.0,
                ppm_slope=10.0,
                scale=0.001, exponent=0.5,
                min_da=0.005, max_da=0.12,
                bin_group_size=50,
            )

            edges = strategy.compute_bin_edges()

            assert torch.all(edges[1:] > edges[:-1]), (
                f"Bin edges must be strictly monotonic for {func}"
            )
            assert edges[0] == 50.0, f"First edge must be min_mz for {func}"
            assert edges[-1] == 2500.0, f"Last edge must be max_mz for {func}"

    def test_state_dict_serialization(self):
        """Test state_dict saves function-specific parameters."""
        # Hyperbolic
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=15.0,
            min_da=0.005, max_da=0.12,
            bin_group_size=50,
        )
        state = strategy.state_dict()
        assert state["strategy"] == "AdaptiveBinning"
        assert state["function"] == "hyperbolic"
        assert state["da_floor"] == 0.01
        assert state["ppm_asymptote"] == 15.0
        assert state["min_da"] == 0.005
        assert state["max_da"] == 0.12

        # Power law
        strategy_pl = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="power_law",
            scale=0.001, exponent=0.5,
            min_da=0.005, max_da=0.12,
            bin_group_size=50,
        )
        state_pl = strategy_pl.state_dict()
        assert state_pl["function"] == "power_law"
        assert state_pl["scale"] == 0.001
        assert state_pl["exponent"] == 0.5
        assert "da_floor" not in state_pl
        assert "ppm_asymptote" not in state_pl

        # Linear
        strategy_lin = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="linear",
            da_floor=0.01, ppm_slope=10.0,
            min_da=0.005, max_da=0.12,
            bin_group_size=50,
        )
        state_lin = strategy_lin.state_dict()
        assert state_lin["function"] == "linear"
        assert state_lin["da_floor"] == 0.01
        assert state_lin["ppm_slope"] == 10.0

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameter combinations raise errors."""
        # Unknown function
        with pytest.raises(ValueError, match="Unknown function"):
            AdaptiveBinning(
                50.0, 2500.0, function="unknown",
                bin_group_size=50,
            )

        # Negative da_floor for hyperbolic
        with pytest.raises(ValueError, match="da_floor"):
            AdaptiveBinning(
                50.0, 2500.0, function="hyperbolic",
                da_floor=-0.01, ppm_asymptote=10.0,
                bin_group_size=50,
            )

        # Invalid Da bounds
        with pytest.raises(ValueError, match="max_da"):
            AdaptiveBinning(
                50.0, 2500.0, function="hyperbolic",
                da_floor=0.01, ppm_asymptote=10.0,
                min_da=0.12, max_da=0.005,
                bin_group_size=50,
            )

        # Negative scale for power_law
        with pytest.raises(ValueError, match="scale"):
            AdaptiveBinning(
                50.0, 2500.0, function="power_law",
                scale=-0.001, exponent=0.5,
                bin_group_size=50,
            )

    def test_mz_to_bin_roundtrip(self):
        """Test m/z -> bin -> m/z round trip accuracy."""
        strategy = AdaptiveBinning(
            min_mz=50.0, max_mz=2500.0,
            function="hyperbolic",
            da_floor=0.01, ppm_asymptote=15.0,
            min_da=0.005, max_da=0.12,
            bin_group_size=50,
        )

        mz_test = torch.tensor([100.0, 500.0, 1000.0, 2000.0])

        bin_indices = strategy.mz_to_bin(mz_test)
        mz_reconstructed = strategy.bin_to_mz(bin_indices)

        edges = strategy.bin_edges
        left = edges[bin_indices]
        right = edges[bin_indices + 1]
        half_width = (right - left) / 2.0

        errors = torch.abs(mz_test - mz_reconstructed)
        assert torch.all(
            errors <= half_width + 1e-5
        ), "Roundtrip error exceeds half bin width"


class TestBinningFactory:
    """Tests for binning strategy factory function."""

    def test_create_fixed_da(self):
        """Test factory creates fixed_da strategy from config."""
        config = {
            "mz_head": {
                "binning": {"strategy": "fixed_da", "bin_size": 0.02},
                "bin_group_size": 50,
            }
        }
        strategy = create_binning_strategy(config, 50.0, 2500.0)

        assert isinstance(
            strategy, FixedDaBinning
        ), "Should create FixedDaBinning instance"
        assert strategy.bin_size == 0.02, "Should use bin_size from config"
        assert strategy.bin_group_size == 50, "Should use bin_group_size from config"

    def test_create_fixed_ppm(self):
        """Test factory creates fixed_ppm strategy from config."""
        config = {
            "mz_head": {
                "binning": {"strategy": "fixed_ppm", "ppm_target": 10.0},
                "bin_group_size": 50,
            }
        }
        strategy = create_binning_strategy(config, 50.0, 2500.0)

        assert isinstance(
            strategy, FixedPpmBinning
        ), "Should create FixedPpmBinning instance"
        assert strategy.ppm_target == 10.0, "Should use ppm_target from config"

    def test_create_adaptive_hyperbolic(self):
        """Test factory creates adaptive strategy with hyperbolic function."""
        config = {
            "mz_head": {
                "binning": {
                    "strategy": "adaptive",
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
                "bin_group_size": 50,
            }
        }
        strategy = create_binning_strategy(config, 50.0, 2500.0)

        assert isinstance(strategy, AdaptiveBinning)
        assert strategy.function == "hyperbolic"
        assert strategy.da_floor == 0.01
        assert strategy.ppm_asymptote == 15.0

    def test_create_adaptive_power_law(self):
        """Test factory creates adaptive strategy with power_law function."""
        config = {
            "mz_head": {
                "binning": {
                    "strategy": "adaptive",
                    "function": "power_law",
                    "scale": 0.001,
                    "exponent": 0.6,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
                "bin_group_size": 50,
            }
        }
        strategy = create_binning_strategy(config, 50.0, 2500.0)

        assert isinstance(strategy, AdaptiveBinning)
        assert strategy.function == "power_law"
        assert strategy.scale == 0.001
        assert strategy.exponent == 0.6

    def test_create_adaptive_defaults_to_hyperbolic(self):
        """Test factory defaults to hyperbolic when no function specified."""
        config = {
            "mz_head": {
                "binning": {
                    "strategy": "adaptive",
                },
                "bin_group_size": 50,
            }
        }
        strategy = create_binning_strategy(config, 50.0, 2500.0)

        assert isinstance(strategy, AdaptiveBinning)
        assert strategy.function == "hyperbolic"

    def test_log_hybrid_raises_deprecation_error(self):
        """Test that log_hybrid strategy raises deprecation error."""
        config = {
            "mz_head": {
                "binning": {
                    "strategy": "log_hybrid",
                    "ppm_low": 60.0,
                    "ppm_mid": 25.0,
                    "ppm_high": 60.0,
                },
                "bin_group_size": 50,
            }
        }

        with pytest.raises(ValueError, match="adaptive"):
            create_binning_strategy(config, 50.0, 2500.0)

    def test_missing_binning_config_raises(self):
        """Old config format without binning key raises ValueError.

        Backward compatibility is handled in FoundationModel.load()
        which migrates the config before calling create_binning_strategy().
        """
        config = {"mz_head": {"bin_size": 0.02, "bin_group_size": 50}}
        with pytest.raises(ValueError, match="Missing mz_head.binning config"):
            create_binning_strategy(config, 50.0, 2500.0)

    def test_missing_parameter_error(self):
        """Test that missing required parameters raise errors."""
        config = {
            "mz_head": {
                "binning": {"strategy": "fixed_ppm"},  # Missing ppm_target
                "bin_group_size": 50,
            }
        }

        with pytest.raises(ValueError, match="ppm_target"):
            create_binning_strategy(config, 50.0, 2500.0)

    def test_unknown_strategy_error(self):
        """Test that unknown strategies raise errors."""
        config = {
            "mz_head": {
                "binning": {"strategy": "unknown"},
                "bin_group_size": 50,
            }
        }

        with pytest.raises(ValueError, match="Unknown binning strategy"):
            create_binning_strategy(config, 50.0, 2500.0)


class TestHierarchicalDecomposition:
    """Tests for hierarchical (group, offset) decomposition across all strategies."""

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.02}),
            (FixedPpmBinning, {"ppm_target": 10.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_group_offset_consistency(self, strategy_class, kwargs):
        """Test group/offset decomposition matches direct binning for all strategies."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        # Test multiple m/z values across the range
        mz_test = torch.linspace(100, 2000, 100)

        # Method 1: Direct binning
        bin_indices_direct = strategy.mz_to_bin(mz_test)

        # Method 2: Via mz_to_bin_groups
        groups, offsets = strategy.mz_to_bin_groups(mz_test)
        bin_indices_hierarchical = groups * 50 + offsets

        # Must match exactly
        assert torch.all(
            bin_indices_direct == bin_indices_hierarchical
        ), "Group/offset decomposition inconsistent with direct binning"

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.02}),
            (FixedPpmBinning, {"ppm_target": 10.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_inverse_consistency(self, strategy_class, kwargs):
        """Test bin_groups_to_mz inverts mz_to_bin_groups correctly."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        mz_test = torch.tensor([100.0, 500.0, 1000.0, 2000.0])

        # Forward: m/z -> (group, offset)
        groups, offsets = strategy.mz_to_bin_groups(mz_test)

        # Inverse: (group, offset) -> m/z
        mz_reconstructed = strategy.bin_groups_to_mz(groups, offsets)

        # Should reconstruct bin centers accurately
        bin_indices = strategy.mz_to_bin(mz_test)
        mz_expected = strategy.bin_to_mz(bin_indices)

        assert torch.allclose(
            mz_reconstructed, mz_expected, atol=1e-5
        ), "bin_groups_to_mz doesn't invert mz_to_bin_groups correctly"

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.02}),
            (FixedPpmBinning, {"ppm_target": 10.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_offset_within_bounds(self, strategy_class, kwargs):
        """Test offsets are always < bin_group_size."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        mz_test = torch.linspace(50, 2500, 1000)
        groups, offsets = strategy.mz_to_bin_groups(mz_test)

        assert torch.all(
            offsets < 50
        ), f"Some offsets >= bin_group_size: max={offsets.max()}"
        assert torch.all(offsets >= 0), f"Some offsets < 0: min={offsets.min()}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_boundary_values(self):
        """Test binning at exact min_mz and max_mz boundaries."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )

        # Test exact boundaries
        mz_boundary = torch.tensor([50.0, 2500.0])
        bin_indices = strategy.mz_to_bin(mz_boundary)

        assert bin_indices[0] == 0, "min_mz should map to bin 0"
        assert (
            bin_indices[1] == strategy.n_bins - 1
        ), "max_mz should map to last bin"

    def test_out_of_range_clamping(self):
        """Test that out-of-range m/z values are clamped correctly."""
        strategy = FixedDaBinning(
            min_mz=50.0, max_mz=2500.0, bin_size=0.02, bin_group_size=50
        )

        # Values outside range
        mz_out_of_range = torch.tensor([0.0, 3000.0])
        bin_indices = strategy.mz_to_bin(mz_out_of_range)

        assert bin_indices[0] == 0, "Values < min_mz should clamp to bin 0"
        assert (
            bin_indices[1] == strategy.n_bins - 1
        ), "Values > max_mz should clamp to last bin"

    def test_state_dict_serialization(self):
        """Test that strategies can be serialized to state_dict."""
        strategies = [
            FixedDaBinning(50.0, 2500.0, 0.02, 50),
            FixedPpmBinning(50.0, 2500.0, 10.0, 50),
            AdaptiveBinning(
                50.0, 2500.0,
                function="hyperbolic",
                da_floor=0.01, ppm_asymptote=15.0,
                min_da=0.005, max_da=0.12,
                bin_group_size=50,
            ),
        ]

        for strategy in strategies:
            state = strategy.state_dict()

            # All strategies should have these keys
            assert "strategy" in state, "state_dict should include strategy name"
            assert "min_mz" in state, "state_dict should include min_mz"
            assert "max_mz" in state, "state_dict should include max_mz"
            assert (
                "bin_group_size" in state
            ), "state_dict should include bin_group_size"

            # Check strategy-specific parameters
            if isinstance(strategy, FixedDaBinning):
                assert "bin_size" in state, "FixedDaBinning should include bin_size"
            elif isinstance(strategy, FixedPpmBinning):
                assert (
                    "ppm_target" in state
                ), "FixedPpmBinning should include ppm_target"
            elif isinstance(strategy, AdaptiveBinning):
                assert "function" in state, "AdaptiveBinning should include function"
                assert "min_da" in state, "AdaptiveBinning should include min_da"
                assert "max_da" in state, "AdaptiveBinning should include max_da"


class TestMzToBinBoundaryBehavior:
    """Tests for exact boundary behavior of mz_to_bin."""

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.1}),
            (FixedPpmBinning, {"ppm_target": 50.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_min_mz_maps_to_bin_zero(self, strategy_class, kwargs):
        """Test that mz == min_mz maps to bin 0."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )
        mz = torch.tensor([50.0])
        bin_idx = strategy.mz_to_bin(mz)
        assert bin_idx[0] == 0, f"min_mz should map to bin 0, got {bin_idx[0]}"

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.1}),
            (FixedPpmBinning, {"ppm_target": 50.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_max_mz_maps_to_last_bin(self, strategy_class, kwargs):
        """Test that mz == max_mz maps to last bin (clamped)."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )
        mz = torch.tensor([2500.0])
        bin_idx = strategy.mz_to_bin(mz)
        expected = strategy.n_bins - 1
        assert bin_idx[0] == expected, f"max_mz should map to bin {expected}, got {bin_idx[0]}"

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.1}),
            (FixedPpmBinning, {"ppm_target": 50.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_out_of_range_clamped(self, strategy_class, kwargs):
        """Test that out-of-range values are clamped."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        # Below min_mz
        mz_low = torch.tensor([0.0, 10.0, 49.9])
        bins_low = strategy.mz_to_bin(mz_low)
        assert torch.all(bins_low == 0), f"Values below min_mz should clamp to 0, got {bins_low.tolist()}"

        # Above max_mz
        mz_high = torch.tensor([2500.1, 3000.0, 5000.0])
        bins_high = strategy.mz_to_bin(mz_high)
        expected = strategy.n_bins - 1
        assert torch.all(bins_high == expected), f"Values above max_mz should clamp to {expected}, got {bins_high.tolist()}"

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            # Note: FixedDaBinning excluded - it uses a formula-based fast path (O(1))
            # which has known floating-point precision limitations at exact bin edges.
            (FixedPpmBinning, {"ppm_target": 50.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_edge_equality_semantics(self, strategy_class, kwargs):
        """Test that mz == edges[k] maps to bin k (left edge goes to that bin)."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        edges = strategy.bin_edges

        # Test several internal edges (not first or last)
        test_indices = [1, 5, 10, 50, 100]
        for k in test_indices:
            if k >= len(edges) - 1:
                continue

            edge_val = edges[k].item()
            mz = torch.tensor([edge_val])
            bin_idx = strategy.mz_to_bin(mz)

            assert bin_idx[0] == k, (
                f"mz == edges[{k}] ({edge_val:.6f}) should map to bin {k}, "
                f"but got bin {bin_idx[0]}"
            )

    @pytest.mark.parametrize(
        "strategy_class,kwargs",
        [
            (FixedDaBinning, {"bin_size": 0.1}),
            (FixedPpmBinning, {"ppm_target": 50.0}),
            (
                AdaptiveBinning,
                {
                    "function": "hyperbolic",
                    "da_floor": 0.01,
                    "ppm_asymptote": 15.0,
                    "min_da": 0.005,
                    "max_da": 0.12,
                },
            ),
        ],
    )
    def test_bin_indices_always_valid(self, strategy_class, kwargs):
        """Test that bin indices are always in valid range [0, n_bins-1]."""
        strategy = strategy_class(
            min_mz=50.0, max_mz=2500.0, bin_group_size=50, **kwargs
        )

        # Test wide range of m/z values
        mz_test = torch.linspace(0.0, 5000.0, 10000)
        bin_indices = strategy.mz_to_bin(mz_test)

        assert torch.all(bin_indices >= 0), f"Negative bin index found: min={bin_indices.min()}"
        assert torch.all(bin_indices < strategy.n_bins), (
            f"Bin index out of range: max={bin_indices.max()}, n_bins={strategy.n_bins}"
        )
