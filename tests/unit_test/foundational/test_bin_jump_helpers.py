#!/usr/bin/env python
"""Unit tests for bin-jump analysis helper methods."""

import sys
import pytest
from pathlib import Path
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instanovo_fm.data.spectrum_analyser import SpectrumAnalyser
from instanovo_fm.utils.peak_classification import extract_fragment_position


@pytest.fixture
def analyser():
    """Create a SpectrumAnalyser instance for testing."""
    config = DictConfig({
        "model": {
            "n_peaks": 200,
            "min_mz": 50.0,
            "max_mz": 2500.0,
            "min_intensity": 0.01,
        },
        "analysis": {
            "enable_mass_error_analysis": True,
            "enable_bin_jump_analysis": True,
            "min_ion_observations": 2,
            "bin_jump_threshold": 0.10,
            "mz_range_thresholds": {
                "low": 200,
                "mid": 800,
                "high": 1500,
            }
        },
        "residues": {
            "residues": {}
        },
        "dataset": {}
    })

    return SpectrumAnalyser(config, output_dir="/tmp/test_bin_jump")


class TestExtractFragmentPosition:
    """Test cases for extract_fragment_position().

    The helper was extracted from SpectrumAnalyser into
    instanovo_fm.utils.peak_classification as a module-level function;
    the assertions below are unchanged.
    """

    def test_standard_annotations(self, analyser):
        """Test position extraction from standard annotations."""
        assert extract_fragment_position("b3+") == 3
        assert extract_fragment_position("y5++") == 5
        assert extract_fragment_position("b10+") == 10
        assert extract_fragment_position("y15++") == 15
        assert extract_fragment_position("c7+") == 7
        assert extract_fragment_position("z9+") == 9

    def test_annotations_with_losses(self, analyser):
        """Test position extraction from annotations with neutral losses."""
        assert extract_fragment_position("b7-H2O") == 7
        assert extract_fragment_position("y12-NH3") == 12
        assert extract_fragment_position("b3-H2O+") == 3
        assert extract_fragment_position("y5-NH3++") == 5

    def test_annotations_with_isotopes(self, analyser):
        """Test position extraction from annotations with isotope markers."""
        assert extract_fragment_position("y12[+1]") == 12
        assert extract_fragment_position("b5[+2]") == 5
        assert extract_fragment_position("y8[+1]++") == 8

    def test_precursor_ions(self, analyser):
        """Test that precursor ions return -1."""
        assert extract_fragment_position("M+2H") == -1
        assert extract_fragment_position("M+3H") == -1
        assert extract_fragment_position("M-H2O") == -1

    def test_invalid_annotations(self, analyser):
        """Test that invalid annotations return -1."""
        assert extract_fragment_position("") == -1
        assert extract_fragment_position(None) == -1
        assert extract_fragment_position("unknown") == -1
        assert extract_fragment_position("xyz") == -1
        assert extract_fragment_position("123") == -1

    def test_case_insensitivity(self, analyser):
        """Test that position extraction is case insensitive."""
        assert extract_fragment_position("B3+") == 3
        assert extract_fragment_position("Y5++") == 5
        assert extract_fragment_position("b3+") == 3

    def test_complex_annotations(self, analyser):
        """Test position extraction from complex annotations."""
        assert extract_fragment_position("y12[loss]-H2O++") == 12
        assert extract_fragment_position("b7-H2O[+1]+") == 7


class TestBinJumpConfiguration:
    """Test configuration parameters for bin-jump analysis."""

    def test_default_config_values(self, analyser):
        """Test that default configuration values are loaded correctly."""
        assert analyser.enable_bin_jump_analysis is True
        assert analyser.min_ion_observations == 2
        assert analyser.bin_jump_threshold == 0.10

    def test_custom_config_values(self):
        """Test that custom configuration values are respected."""
        config = DictConfig({
            "model": {
                "n_peaks": 200,
                "min_mz": 50.0,
                "max_mz": 2500.0,
                "min_intensity": 0.01,
            },
            "analysis": {
                "enable_bin_jump_analysis": False,
                "min_ion_observations": 5,
                "bin_jump_threshold": 0.05,
            },
            "residues": {
                "residues": {}
            },
            "dataset": {}
        })

        analyser = SpectrumAnalyser(config, output_dir="/tmp/test_bin_jump_custom")

        assert analyser.enable_bin_jump_analysis is False
        assert analyser.min_ion_observations == 5
        assert analyser.bin_jump_threshold == 0.05


class TestPartDConfiguration:
    """Test Part D extended analysis configuration parameters."""

    def test_part_d_default_values(self):
        """Test that Part D default configuration values are loaded correctly."""
        config = DictConfig({
            "model": {
                "n_peaks": 200,
                "min_mz": 50.0,
                "max_mz": 2500.0,
                "min_intensity": 0.01,
            },
            "analysis": {},
            "residues": {
                "residues": {}
            },
            "dataset": {}
        })

        analyser = SpectrumAnalyser(config, output_dir="/tmp/test_part_d_default")

        # Part D defaults
        assert analyser.enable_collision_analysis is True
        assert analyser.enable_stratified_analysis is True
        assert analyser.enable_soft_label_evaluation is True
        assert analyser.collision_weight == 0.4
        assert analyser.boundary_threshold == 0.4
        assert analyser.soft_label_boundary_threshold == 0.3
        assert analyser.soft_label_adjacent_threshold == 0.7

    def test_part_d_custom_values(self):
        """Test that Part D custom configuration values are respected."""
        config = DictConfig({
            "model": {
                "n_peaks": 200,
                "min_mz": 50.0,
                "max_mz": 2500.0,
                "min_intensity": 0.01,
            },
            "analysis": {
                "enable_collision_analysis": False,
                "enable_stratified_analysis": False,
                "enable_soft_label_evaluation": False,
                "collision_weight": 0.6,
                "boundary_threshold": 0.3,
                "soft_label_recommendation_threshold": {
                    "boundary_fraction": 0.4,
                    "adjacent_jump_fraction": 0.8,
                }
            },
            "residues": {
                "residues": {}
            },
            "dataset": {}
        })

        analyser = SpectrumAnalyser(config, output_dir="/tmp/test_part_d_custom")

        assert analyser.enable_collision_analysis is False
        assert analyser.enable_stratified_analysis is False
        assert analyser.enable_soft_label_evaluation is False
        assert analyser.collision_weight == 0.6
        assert analyser.boundary_threshold == 0.3
        assert analyser.soft_label_boundary_threshold == 0.4
        assert analyser.soft_label_adjacent_threshold == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
