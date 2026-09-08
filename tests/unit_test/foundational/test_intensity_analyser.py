#!/usr/bin/env python
"""
Unit tests for IntensityAnalyser module.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from instanovo_fm.data.intensity_analyser import IntensityAnalyser


@pytest.fixture
def default_config():
    """Create default configuration for tests."""
    config = {
        "model": {
            "max_mz": 2500.0,
            "min_mz": 50.0,
            "min_intensity": 0.01,
        },
        "analysis": {
            "enable_intensity_analysis": True,
            "intensity_bins": 50,
            "intensity_log_scale": True,
            "compute_skewness_kurtosis": True,
            "mz_range_boundaries": {
                "immonium_internal": [0, 200],
                "core_fragment": [200, 800],
                "extended_fragment": [800, 1500],
                "high_mass_fragment": [1500, 2500],
            }
        }
    }
    return DictConfig(config)


@pytest.fixture
def intensity_analyser(default_config):
    """Create IntensityAnalyser instance with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        analyser = IntensityAnalyser(default_config, output_dir=Path(tmpdir))
        yield analyser


class TestIntensityAnalyserInit:
    """Test IntensityAnalyser initialization."""

    def test_init_with_config(self, default_config):
        """Test initialization with configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyser = IntensityAnalyser(default_config, output_dir=Path(tmpdir))

            assert analyser.max_mz == 2500.0
            assert analyser.min_mz == 50.0
            assert analyser.min_intensity == 0.01
            assert analyser.intensity_bins == 50
            assert analyser.intensity_log_scale is True
            assert analyser.compute_skewness_kurtosis is True
            assert analyser.output_dir == Path(tmpdir)
            assert analyser.output_dir.exists()

    def test_init_creates_output_directory(self, default_config):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "intensity_test"
            analyser = IntensityAnalyser(default_config, output_dir=output_dir)

            assert analyser.output_dir.exists()
            assert analyser.output_dir == output_dir

    def test_mz_range_boundaries_loaded(self, intensity_analyser):
        """Test that m/z range boundaries are loaded correctly."""
        assert "immonium_internal" in intensity_analyser.mz_range_boundaries
        assert "core_fragment" in intensity_analyser.mz_range_boundaries
        assert "extended_fragment" in intensity_analyser.mz_range_boundaries
        assert "high_mass_fragment" in intensity_analyser.mz_range_boundaries

        # Check tuple format
        assert intensity_analyser.mz_range_boundaries["immonium_internal"] == (0, 200)
        assert intensity_analyser.mz_range_boundaries["core_fragment"] == (200, 800)


class TestClassifyMzRange:
    """Test m/z range classification."""

    def test_classify_mz_range_low(self, intensity_analyser):
        """Test classification of low m/z values."""
        assert intensity_analyser._classify_mz_range(100.0) == "immonium_internal"
        assert intensity_analyser._classify_mz_range(50.0) == "immonium_internal"
        assert intensity_analyser._classify_mz_range(199.9) == "immonium_internal"

    def test_classify_mz_range_mid(self, intensity_analyser):
        """Test classification of mid m/z values."""
        assert intensity_analyser._classify_mz_range(200.0) == "core_fragment"
        assert intensity_analyser._classify_mz_range(500.0) == "core_fragment"
        assert intensity_analyser._classify_mz_range(799.9) == "core_fragment"

    def test_classify_mz_range_high(self, intensity_analyser):
        """Test classification of high m/z values."""
        assert intensity_analyser._classify_mz_range(800.0) == "extended_fragment"
        assert intensity_analyser._classify_mz_range(1200.0) == "extended_fragment"
        assert intensity_analyser._classify_mz_range(1499.9) == "extended_fragment"

    def test_classify_mz_range_very_high(self, intensity_analyser):
        """Test classification of very high m/z values."""
        assert intensity_analyser._classify_mz_range(1500.0) == "high_mass_fragment"
        assert intensity_analyser._classify_mz_range(2000.0) == "high_mass_fragment"
        assert intensity_analyser._classify_mz_range(2500.0) == "high_mass_fragment"


class TestAnalyzeSpectrum:
    """Test per-spectrum analysis."""

    def test_analyze_spectrum_basic(self, intensity_analyser):
        """Test basic spectrum analysis returns expected fields."""
        # Create synthetic spectrum
        valid_mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        valid_intensity = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        metadata = {
            "frag_type": "HCD",
            "search_instrument": "Orbitrap",
            "precursor_charge": 3,
        }

        result = intensity_analyser.analyze_spectrum(
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata=metadata
        )

        # Check required fields
        assert "n_peaks" in result
        assert "mean_intensity" in result
        assert "median_intensity" in result
        assert "std_intensity" in result
        assert "min_intensity" in result
        assert "max_intensity" in result
        assert "dynamic_range" in result
        assert "intensity_cv" in result
        assert "raw_intensity_records" in result

        # Check values
        assert result["n_peaks"] == 5
        assert result["mean_intensity"] == pytest.approx(0.3, abs=0.01)
        assert result["median_intensity"] == pytest.approx(0.3, abs=0.01)
        assert result["min_intensity"] == 0.1
        assert result["max_intensity"] == 0.5
        assert result["dynamic_range"] == pytest.approx(5.0, abs=0.01)

    def test_analyze_spectrum_empty(self, intensity_analyser):
        """Test handling of empty spectrum."""
        valid_mz = np.array([])
        valid_intensity = np.array([])
        metadata = {"frag_type": "HCD", "search_instrument": "Orbitrap", "precursor_charge": 3}

        result = intensity_analyser.analyze_spectrum(
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata=metadata
        )

        assert result["n_peaks"] == 0
        assert "error" in result

    def test_analyze_spectrum_creates_records(self, intensity_analyser):
        """Test that per-peak records are created as dict-of-arrays (columnar format)."""
        valid_mz = np.array([100.0, 500.0, 1200.0])
        valid_intensity = np.array([0.2, 0.5, 0.3])
        metadata = {"frag_type": "HCD", "search_instrument": "Orbitrap", "precursor_charge": 2}

        result = intensity_analyser.analyze_spectrum(
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata=metadata
        )

        records = result["raw_intensity_records"]
        # Should be dict-of-arrays, not list-of-dicts
        assert isinstance(records, dict)
        assert "intensity" in records
        assert "mz" in records
        assert "mz_range" in records
        assert "frag_type" in records
        assert "instrument" in records
        assert "charge" in records

        # All arrays should have length 3
        for key in ["intensity", "mz", "mz_range", "frag_type", "instrument", "charge"]:
            assert len(records[key]) == 3

        # Check first peak values
        assert records["intensity"][0] == pytest.approx(0.2)
        assert records["mz"][0] == pytest.approx(100.0)
        assert records["mz_range"][0] == "immonium_internal"
        assert records["frag_type"][0] == "HCD"
        assert records["instrument"][0] == "Orbitrap"
        assert records["charge"][0] == 2

        # Check m/z range classification for remaining peaks
        assert records["mz_range"][1] == "core_fragment"
        assert records["mz_range"][2] == "extended_fragment"

    def test_coefficient_of_variation(self, intensity_analyser):
        """Test CV computation."""
        valid_mz = np.array([100.0, 200.0])
        valid_intensity = np.array([0.1, 0.5])  # mean=0.3, std=0.2
        metadata = {"frag_type": "HCD", "search_instrument": "Orbitrap", "precursor_charge": 2}

        result = intensity_analyser.analyze_spectrum(
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata=metadata
        )

        expected_cv = 0.2 / 0.3  # std / mean
        assert result["intensity_cv"] == pytest.approx(expected_cv, abs=0.01)

    def test_dynamic_range(self, intensity_analyser):
        """Test dynamic range computation."""
        valid_mz = np.array([100.0, 200.0, 300.0])
        valid_intensity = np.array([0.01, 0.5, 1.0])  # min=0.01, max=1.0
        metadata = {"frag_type": "HCD", "search_instrument": "Orbitrap", "precursor_charge": 2}

        result = intensity_analyser.analyze_spectrum(
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata=metadata
        )

        expected_dr = 1.0 / 0.01  # max / min = 100
        assert result["dynamic_range"] == pytest.approx(expected_dr, abs=0.1)


class TestCalculateOverallStats:
    """Test overall statistics calculation."""

    def test_overall_stats_basic(self, intensity_analyser):
        """Test basic overall statistics."""
        # Create synthetic dataframe
        data = {
            "intensity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "mz": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "mz_range": ["core_fragment"] * 10,
            "frag_type": ["HCD"] * 10,
            "instrument": ["Orbitrap"] * 10,
            "charge": [2] * 10,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_overall_intensity_stats(df)

        assert stats["n_peaks"] == 10
        assert stats["mean"] == pytest.approx(0.55, abs=0.01)
        assert stats["median"] == pytest.approx(0.55, abs=0.01)
        assert stats["min"] == 0.1
        assert stats["max"] == 1.0
        assert "percentiles" in stats
        assert "p5" in stats["percentiles"]
        assert "p95" in stats["percentiles"]

    def test_percentile_calculation(self, intensity_analyser):
        """Test percentile calculation."""
        data = {
            "intensity": np.linspace(0, 1, 101),  # 0, 0.01, 0.02, ..., 1.0
            "mz": np.linspace(100, 1000, 101),
            "mz_range": ["core_fragment"] * 101,
            "frag_type": ["HCD"] * 101,
            "instrument": ["Orbitrap"] * 101,
            "charge": [2] * 101,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_overall_intensity_stats(df)

        # Check percentiles
        assert stats["percentiles"]["p5"] == pytest.approx(0.05, abs=0.01)
        assert stats["percentiles"]["p50"] == pytest.approx(0.50, abs=0.01)
        assert stats["percentiles"]["p95"] == pytest.approx(0.95, abs=0.01)

    def test_intensity_concentration(self, intensity_analyser):
        """Test intensity concentration calculation."""
        # Create highly concentrated distribution (one peak has most intensity)
        intensities = [1.0] + [0.01] * 99  # Top 1% has 91% of total intensity
        data = {
            "intensity": intensities,
            "mz": np.linspace(100, 1000, 100),
            "mz_range": ["core_fragment"] * 100,
            "frag_type": ["HCD"] * 100,
            "instrument": ["Orbitrap"] * 100,
            "charge": [2] * 100,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_overall_intensity_stats(df)

        concentration = stats["intensity_concentration"]
        # Top 1% should have most of the intensity
        assert concentration["top_1_percent_fraction"] > 0.5
        assert concentration["top_10_percent_fraction"] > concentration["top_1_percent_fraction"]


class TestCalculateStratifiedStats:
    """Test stratified statistics calculation."""

    def test_stratified_stats_by_frag_type(self, intensity_analyser):
        """Test stratification by fragmentation type."""
        data = {
            "intensity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2,
            "mz": [100] * 20,
            "mz_range": ["core_fragment"] * 20,
            "frag_type": ["HCD"] * 10 + ["CID"] * 10,
            "instrument": ["Orbitrap"] * 20,
            "charge": [2] * 20,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_stratified_stats(df, "frag_type")

        assert "HCD" in stats
        assert "CID" in stats
        assert stats["HCD"]["n_peaks"] == 10
        assert stats["CID"]["n_peaks"] == 10
        assert "mean" in stats["HCD"]
        assert "cv" in stats["HCD"]
        assert "percentiles" in stats["HCD"]

    def test_stratified_stats_skips_small_groups(self, intensity_analyser):
        """Test that small groups (<10 peaks) are skipped."""
        data = {
            "intensity": [0.5] * 5 + [0.6] * 15,
            "mz": [100] * 20,
            "mz_range": ["core_fragment"] * 20,
            "frag_type": ["HCD"] * 5 + ["CID"] * 15,  # HCD has only 5 peaks
            "instrument": ["Orbitrap"] * 20,
            "charge": [2] * 20,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_stratified_stats(df, "frag_type")

        assert "HCD" not in stats  # Should be skipped (< 10 peaks)
        assert "CID" in stats

    def test_stratified_stats_unknown_filtered(self, intensity_analyser):
        """Test that 'unknown' values are filtered out."""
        data = {
            "intensity": [0.5] * 30,
            "mz": [100] * 30,
            "mz_range": ["core_fragment"] * 30,
            "frag_type": ["HCD"] * 10 + ["unknown"] * 10 + ["CID"] * 10,
            "instrument": ["Orbitrap"] * 30,
            "charge": [2] * 30,
        }
        df = pd.DataFrame(data)

        stats = intensity_analyser._calculate_stratified_stats(df, "frag_type")

        assert "unknown" not in stats
        assert "HCD" in stats
        assert "CID" in stats


class TestAggregateResults:
    """Test result aggregation."""

    def test_aggregate_results_dict_of_arrays(self, intensity_analyser):
        """Test aggregation with new dict-of-arrays format."""
        per_spectrum_results = []
        for i in range(10):
            n = 5
            result = {
                "n_peaks": n,
                "mean_intensity": 0.5,
                "dynamic_range": 10.0,
                "intensity_cv": 0.4,
                "raw_intensity_records": {
                    "intensity": np.full(n, 0.1 + i * 0.01),
                    "mz": np.full(n, 100.0 + i * 10),
                    "mz_range": np.full(n, "core_fragment", dtype=object),
                    "frag_type": np.full(n, "HCD", dtype=object),
                    "instrument": np.full(n, "Orbitrap", dtype=object),
                    "charge": np.full(n, 2, dtype=object),
                },
            }
            per_spectrum_results.append(result)

        aggregated = intensity_analyser.aggregate_results(per_spectrum_results)

        assert "overall_intensity_stats" in aggregated
        assert "stratified_intensity" in aggregated
        assert "cross_stratified_intensity" in aggregated
        assert "raw_data" in aggregated

        overall = aggregated["overall_intensity_stats"]
        assert overall["n_peaks"] == 50  # 10 spectra * 5 peaks

    def test_aggregate_results_legacy_list_of_dicts(self, intensity_analyser):
        """Test aggregation with legacy list-of-dicts format (backward compat)."""
        per_spectrum_results = []
        for i in range(10):
            result = {
                "n_peaks": 5,
                "mean_intensity": 0.5,
                "raw_intensity_records": [
                    {
                        "intensity": 0.1 + i * 0.01,
                        "mz": 100.0 + i * 10,
                        "mz_range": "core_fragment",
                        "frag_type": "HCD",
                        "instrument": "Orbitrap",
                        "charge": 2,
                    }
                    for _ in range(5)
                ]
            }
            per_spectrum_results.append(result)

        aggregated = intensity_analyser.aggregate_results(per_spectrum_results)

        assert "overall_intensity_stats" in aggregated
        overall = aggregated["overall_intensity_stats"]
        assert overall["n_peaks"] == 50

    def test_aggregate_results_empty(self, intensity_analyser):
        """Test aggregation with no data."""
        per_spectrum_results = []

        aggregated = intensity_analyser.aggregate_results(per_spectrum_results)

        assert "error" in aggregated


class TestSaveResults:
    """Test result saving."""

    def test_save_results_creates_files(self, intensity_analyser):
        """Test that save_results creates expected files."""
        # Create minimal results
        intensity_analyser.results = {
            "overall_intensity_stats": {"mean": 0.5, "n_peaks": 100},
            "stratified_intensity": {},
            "cross_stratified_intensity": {},
            "raw_data": pd.DataFrame({
                "intensity": [0.1, 0.2],
                "mz": [100.0, 200.0],
            }),
            "per_spectrum": [
                {"n_peaks": 2, "mean_intensity": 0.15, "metadata": {"frag_type": "HCD", "search_instrument": "Orbitrap"}}
            ]
        }

        intensity_analyser.save_results()

        # Check that files were created
        summary_file = intensity_analyser.output_dir / "intensity_analysis_summary.json"
        per_peak_file = intensity_analyser.output_dir / "intensity_per_peak.csv"
        per_spectrum_file = intensity_analyser.output_dir / "intensity_per_spectrum.csv"

        assert summary_file.exists()
        assert per_peak_file.exists()
        assert per_spectrum_file.exists()

        # Verify JSON is valid
        with open(summary_file) as f:
            data = json.load(f)
            assert "overall_intensity_stats" in data


class TestPrintSummary:
    """Test summary printing."""

    def test_print_summary_no_crash(self, intensity_analyser):
        """Test that print_summary doesn't crash with valid results."""
        intensity_analyser.results = {
            "overall_intensity_stats": {
                "n_peaks": 1000,
                "mean": 0.5,
                "median": 0.4,
                "std": 0.2,
                "coefficient_of_variation": 0.4,
                "dynamic_range": 100.0,
                "percentiles": {"p95": 0.9}
            },
            "stratified_intensity": {
                "by_frag_type": {
                    "HCD": {"mean": 0.5, "cv": 0.4, "n_peaks": 800},
                    "CID": {"mean": 0.45, "cv": 0.45, "n_peaks": 200},
                }
            }
        }

        # Should not raise exception
        intensity_analyser.print_summary()

    def test_print_summary_empty_results(self, intensity_analyser):
        """Test print_summary with empty results."""
        intensity_analyser.results = {}

        # Should not raise exception, just warn
        intensity_analyser.print_summary()


class TestMzRangeVectorized:
    """Test vectorized m/z range classification."""

    def test_vectorized_matches_scalar(self, intensity_analyser):
        """Vectorized classification should match per-element scalar classification."""
        mz_values = np.array([50.0, 150.0, 200.0, 500.0, 800.0, 1200.0, 1500.0, 2000.0, 2499.0])
        vectorized = intensity_analyser._classify_mz_range_vectorized(mz_values)
        scalar = np.array([intensity_analyser._classify_mz_range(m) for m in mz_values])
        np.testing.assert_array_equal(vectorized, scalar)

    def test_vectorized_empty(self, intensity_analyser):
        """Vectorized classification on empty array returns empty."""
        result = intensity_analyser._classify_mz_range_vectorized(np.array([]))
        assert len(result) == 0


class TestBoundaryMaxMz:
    """Test that high_mass_fragment uses max_mz instead of infinity."""

    def test_default_boundary_uses_max_mz(self, default_config):
        """Default high_mass_fragment upper bound should be max_mz, not inf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyser = IntensityAnalyser(default_config, output_dir=Path(tmpdir))
            _, high = analyser.mz_range_boundaries["high_mass_fragment"]
            assert high == 2500.0
            assert high != float("inf")

    def test_mz_at_max_mz_falls_to_default(self, intensity_analyser):
        """m/z exactly at max_mz should fall to the default (last) range."""
        # With boundary [1500, 2500.0), mz=2500.0 is NOT < 2500.0
        # so it falls through to the default (last range)
        result = intensity_analyser._classify_mz_range(2500.0)
        assert result == "high_mass_fragment"


class TestGenerateVisualizations:
    """Crash-safety tests for visualization generation."""

    def _build_results(self, intensity_analyser, n_spectra=20, n_peaks_per=10):
        """Build realistic results dict for visualization tests."""
        per_spectrum_results = []
        for i in range(n_spectra):
            mz = np.linspace(100, 2000, n_peaks_per)
            intensity = np.random.RandomState(i).uniform(0.01, 1.0, n_peaks_per)
            metadata = {
                "frag_type": "HCD" if i % 2 == 0 else "CID",
                "search_instrument": "Orbitrap",
                "precursor_charge": (i % 3) + 2,
            }
            result = intensity_analyser.analyze_spectrum(
                valid_mz=mz,
                valid_intensity=intensity,
                metadata=metadata,
            )
            per_spectrum_results.append(result)

        intensity_analyser.aggregate_results(per_spectrum_results)

    def test_generate_overall_no_crash(self, intensity_analyser):
        """Overall intensity distribution plot should not crash."""
        self._build_results(intensity_analyser)
        intensity_analyser._generate_overall_intensity_distribution_plot()
        output = intensity_analyser.output_dir / "intensity_distribution_overall.png"
        assert output.exists()

    def test_generate_stratified_no_crash(self, intensity_analyser):
        """Stratified intensity comparison plot should not crash."""
        self._build_results(intensity_analyser)
        intensity_analyser._generate_stratified_intensity_plots()
        output = intensity_analyser.output_dir / "intensity_stratified_comparison.png"
        assert output.exists()

    def test_generate_cross_comparison_no_crash(self, intensity_analyser):
        """Cross-comparison plot should not crash."""
        self._build_results(intensity_analyser)
        intensity_analyser._generate_cross_comparison_plots()
        output = intensity_analyser.output_dir / "intensity_cross_comparison.png"
        assert output.exists()

    def test_generate_all_no_crash(self, intensity_analyser):
        """Full generate_visualizations should not crash."""
        self._build_results(intensity_analyser)
        intensity_analyser.generate_visualizations()
        assert (intensity_analyser.output_dir / "intensity_distribution_overall.png").exists()
        assert (intensity_analyser.output_dir / "intensity_stratified_comparison.png").exists()
        assert (intensity_analyser.output_dir / "intensity_cross_comparison.png").exists()

    def test_generate_with_no_results_no_crash(self, intensity_analyser):
        """Visualization with empty results should not crash."""
        intensity_analyser.results = {}
        intensity_analyser.generate_visualizations()

    def test_generate_with_annotations(self, intensity_analyser):
        """Visualization with annotation data should not crash."""
        per_spectrum_results = []
        for i in range(20):
            mz = np.linspace(100, 2000, 10)
            intensity = np.random.RandomState(i).uniform(0.01, 1.0, 10)
            annotations = ["b1", "", "y2", "", "b3", "", "", "y4", "", ""]
            metadata = {
                "frag_type": "HCD",
                "search_instrument": "Orbitrap",
                "precursor_charge": 2,
            }
            result = intensity_analyser.analyze_spectrum(
                valid_mz=mz,
                valid_intensity=intensity,
                metadata=metadata,
                annotations=annotations,
            )
            per_spectrum_results.append(result)

        intensity_analyser.aggregate_results(per_spectrum_results)
        intensity_analyser.generate_visualizations()
        assert (intensity_analyser.output_dir / "intensity_distribution_overall.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
