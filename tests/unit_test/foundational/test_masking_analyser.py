#!/usr/bin/env python
"""Unit tests for MaskingAnalyser module."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from instanovo_fm.data.masking_analyser import MaskingAnalyser


@pytest.fixture
def default_config():
    """Minimal config that MaskingAnalyser and MaskingGapAnalyser need."""
    return DictConfig({
        "model": {
            "max_mz": 2500.0,
            "min_mz": 50.0,
            "min_intensity": 0.01,
            "normalize_mz": True,
            "mz_head": {
                "binning": {"bin_size": 0.02},
                "bin_group_size": 50,
            },
        },
        "analysis": {
            "enable_masking_analysis": True,
            "masking_analysis_n_repeats": 1,
            "masking_strategies": {
                "uniform": {"type": "uniform", "mask_portion": 0.30},
                "thompson": {
                    "type": "thompson",
                    "mask_portion": 0.30,
                    "alpha": 0.5,
                    "beta": 0.5,
                    "kappa": 4.0,
                    "gamma": 0.7,
                },
            },
            "mz_range_boundaries": {
                "immonium_internal": [0, 200],
                "core_fragment": [200, 800],
                "extended_fragment": [800, 1500],
                "high_mass_fragment": [1500, 2500],
            },
            # Synthetic fixtures can't satisfy the production quality gate
            # (5 fragment groups vs. min 7), so disable it in tests that
            # exercise analyser wiring rather than gate behaviour. Tests
            # for the gate itself should override this in their own config.
            "task_configs": {
                "theoretical": {
                    "theoretical_analysis": {"apply_quality_gate_filter": False},
                },
            },
        },
    })


@pytest.fixture
def analyser(default_config):
    """Create MaskingAnalyser with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield MaskingAnalyser(default_config, output_dir=Path(tmpdir))


def _make_synthetic_spectrum(n_peaks: int = 50, max_mz: float = 2500.0):
    """Create a synthetic preprocessed spectrum tensor.

    Returns:
        spectra: [1, n_peaks, 2] — normalised m/z in col 0, intensity in col 1
        spectra_mask: [1, n_peaks] bool — True for padding positions
        valid_mz: 1-D numpy array of non-padded m/z values (Da)
        valid_intensity: 1-D numpy array of non-padded intensities
    """
    n_valid = n_peaks - 10  # 10 padding positions
    mz_da = np.sort(np.random.uniform(100.0, 1500.0, size=n_valid))
    mz_norm = mz_da / max_mz
    intensity = np.random.uniform(0.01, 1.0, size=n_valid).astype(np.float32)

    spectra = torch.zeros(1, n_peaks, 2)
    spectra[0, :n_valid, 0] = torch.from_numpy(mz_norm.astype(np.float32))
    spectra[0, :n_valid, 1] = torch.from_numpy(intensity)

    spectra_mask = torch.ones(1, n_peaks, dtype=torch.bool)
    spectra_mask[0, :n_valid] = False

    return spectra, spectra_mask, mz_da, intensity


class TestMaskingAnalyserInit:
    def test_strategies_loaded(self, analyser):
        assert "uniform" in analyser.strategies
        assert "thompson" in analyser.strategies
        assert len(analyser.strategies) == 2

    def test_gap_analysers_created(self, analyser):
        assert "uniform" in analyser.gap_analysers
        assert "thompson" in analyser.gap_analysers

    def test_output_dirs_created(self, analyser):
        assert analyser.output_dir.exists()
        assert (analyser.output_dir / "uniform").exists()
        assert (analyser.output_dir / "thompson").exists()


class TestApplyStrategy:
    def test_uniform_produces_mask(self, analyser):
        spectra, spectra_mask, _, _ = _make_synthetic_spectrum()
        mask, fallback = analyser._apply_strategy(
            "uniform", analyser.strategies["uniform"], spectra, spectra_mask
        )
        assert mask.shape == (1, spectra.shape[1])
        assert mask.dtype == torch.bool
        assert fallback is None  # uniform has no fallback path
        # Should mask some (but not all) valid peaks
        valid = ~spectra_mask[0]
        n_masked = mask[0][valid].sum().item()
        assert 0 < n_masked < valid.sum().item()

    def test_thompson_produces_mask(self, analyser):
        spectra, spectra_mask, _, _ = _make_synthetic_spectrum()
        mask, fallback = analyser._apply_strategy(
            "thompson", analyser.strategies["thompson"], spectra, spectra_mask
        )
        assert mask.shape == (1, spectra.shape[1])
        assert fallback is None  # thompson has no fallback path
        valid = ~spectra_mask[0]
        n_masked = mask[0][valid].sum().item()
        assert 0 < n_masked < valid.sum().item()

    def test_padding_positions_not_masked(self, analyser):
        spectra, spectra_mask, _, _ = _make_synthetic_spectrum()
        for name, scfg in analyser.strategies.items():
            mask, _ = analyser._apply_strategy(name, scfg, spectra, spectra_mask)
            # Padding positions (spectra_mask=True) should NOT be masked
            padding = spectra_mask[0]
            assert (mask[0][padding] == False).all(), f"Strategy {name} masks padding positions"


class TestAnalyzeSpectrum:
    def test_returns_all_strategies(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        assert "uniform" in result
        assert "thompson" in result

    def test_each_strategy_has_mask_ratio(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        for name in analyser.strategies:
            assert "mask_ratio" in result[name]
            assert 0 < result[name]["mask_ratio"] < 1

    def test_gap_result_present(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        for name in analyser.strategies:
            repeats = result[name]["repeats"]
            assert len(repeats) == 1
            assert "gap_result" in repeats[0]

    def test_theoretical_masking_effect_empty_without_annotation(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
            theoretical_analysis=None,
        )
        for name in analyser.strategies:
            assert result[name]["repeats"][0]["masking_effect"] == {}

    def test_n_repeats(self, default_config):
        """Test multiple repeats per strategy."""
        default_config.analysis.masking_analysis_n_repeats = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MaskingAnalyser(default_config, output_dir=Path(tmpdir))
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            result = a.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
            )
            for name in a.strategies:
                assert len(result[name]["repeats"]) == 3


class TestAggregateResults:
    def test_aggregate_with_multiple_spectra(self, analyser):
        all_results = []
        for _ in range(5):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
            )
            all_results.append(result)

        agg = analyser.aggregate_results(all_results)

        assert "per_strategy" in agg
        assert "comparison" in agg
        assert agg["n_spectra"] == 5
        assert agg["n_strategies"] == 2

        for name in analyser.strategies:
            strat = agg["per_strategy"][name]
            assert "mask_ratio_stats" in strat
            assert "gap_analysis" in strat
            assert strat["mask_ratio_stats"]["mean"] > 0

    def test_behavior_metrics_in_aggregate(self, analyser):
        """Verify behavior metrics (run lengths, curves) are computed."""
        all_results = []
        for _ in range(5):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
            )
            all_results.append(result)

        agg = analyser.aggregate_results(all_results)

        for name in analyser.strategies:
            behavior = agg["per_strategy"][name].get("behavior", {})
            assert "run_lengths" in behavior
            assert len(behavior["run_lengths"]) > 0
            # All run lengths should be positive integers
            assert (behavior["run_lengths"] >= 1).all()

            assert "intensity_mask_curve" in behavior
            imc = behavior["intensity_mask_curve"]
            assert len(imc["mean_rates"]) == 10  # 10 decile bins
            assert imc["n_spectra"] == 5

            assert "spatial_mask_curve" in behavior
            smc = behavior["spatial_mask_curve"]
            assert len(smc["mean_rates"]) == 25  # 25 m/z bins
            assert len(smc["mz_bin_centers"]) == 25

    def test_comparison_tables(self, analyser):
        all_results = []
        for _ in range(3):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
            )
            all_results.append(result)

        agg = analyser.aggregate_results(all_results)
        comparison = agg["comparison"]

        assert "mask_ratio" in comparison
        assert "gap_metrics" in comparison
        assert "annotated_interaction" in comparison
        assert "training_signal" in comparison
        assert len(comparison["mask_ratio"]) == 2  # uniform + thompson

    def test_empty_input(self, analyser):
        agg = analyser.aggregate_results([])
        assert agg == {}


class TestAnnotationDrivenSubset:
    """Gold-standard companion view — restricted to spectra where
    ``signal_aware_fragment`` used the annotation-driven path.

    These tests inject per-repeat ``fallback_used`` flags directly rather
    than going through ``analyze_spectrum`` (which would require a real
    annotation pipeline). The goal is to confirm the aggregation-level
    plumbing: the subset is built, keyed on the reference strategy's
    flag, and stored under the expected keys.
    """

    def _make_record(self, strategies, fallback_used, sigaware_key):
        """Build a minimal per-spectrum result for every configured strategy.

        ``fallback_used`` is applied to whichever strategy has
        ``type == "signal_aware_fragment"`` (addressed by its config key
        ``sigaware_key``). Other strategies get ``None``.
        """
        record: Dict[str, Any] = {
            "_viz_data": {
                "valid_intensity": np.zeros(0, dtype=np.float32),
                "valid_mz": np.zeros(0, dtype=np.float32),
                "n_valid": 0,
                "metadata": {
                    "frag_type": "HCD",
                    "precursor_charge": "2",
                    "search_instrument": "Orbitrap",
                },
            },
        }
        for name in strategies:
            record[name] = {
                "repeats": [
                    {
                        "mask_ratio": 0.30,
                        "gap_result": None,
                        "masking_effect": {},
                        "leakage": None,
                        "mlm_mask_valid": None,
                        "masked_intensity_mean": 0.0,
                        "unmasked_intensity_mean": 0.0,
                        "fallback_used": (
                            fallback_used if name == sigaware_key else None
                        ),
                    }
                ],
                "mask_ratio": 0.30,
            }
        return record

    def test_subset_absent_when_no_signal_aware(self, analyser):
        """The default fixture has no signal_aware_fragment; subset must
        be skipped without raising."""
        records = [
            self._make_record(
                list(analyser.strategies.keys()),
                fallback_used=None,
                sigaware_key="__none__",
            )
            for _ in range(3)
        ]
        agg = analyser.aggregate_results(records)
        assert "per_strategy_annotation_driven" not in agg
        assert "annotation_driven_meta" not in agg

    @pytest.mark.parametrize("sigaware_key", ["signal_aware_fragment", "sigaw_mp30_cap40"])
    def test_subset_populated_with_mixed_flags(self, default_config, sigaware_key):
        """Run the analyser with a signal_aware_fragment strategy configured
        under an arbitrary key name and inject a mix of annotation-driven
        and fallback spectra; confirm the subset is built, counts
        reconcile, and meta records the *key* (so downstream visualisations
        can cross-reference). Parametrising on the key name guards against
        the previous hardcoded-key bug — the analyser now resolves the
        reference by ``type``."""
        cfg = DictConfig({
            **default_config,
            "analysis": {
                **default_config["analysis"],
                "masking_strategies": {
                    **default_config["analysis"]["masking_strategies"],
                    sigaware_key: {
                        "type": "signal_aware_fragment",
                        "mask_portion": 0.30,
                        "min_backbone_coverage": 0.33,
                        "min_fragment_groups": 7,
                        "annotation_ppm": 20.0,
                        "annotation_cid_da_tol": 0.2,
                        "annotation_ion_types": ["b", "y"],
                        "max_total_mask_ratio": 0.40,
                    },
                },
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_analyser = MaskingAnalyser(cfg, output_dir=Path(tmpdir))
            strategy_names = list(sa_analyser.strategies.keys())
            assert sigaware_key in strategy_names

            records = [
                self._make_record(
                    strategy_names, fallback_used=False, sigaware_key=sigaware_key
                )
                for _ in range(4)
            ] + [
                self._make_record(
                    strategy_names, fallback_used=True, sigaware_key=sigaware_key
                )
                for _ in range(2)
            ]
            agg = sa_analyser.aggregate_results(records)

            assert "per_strategy_annotation_driven" in agg
            assert "annotation_driven_meta" in agg
            meta = agg["annotation_driven_meta"]
            assert meta["n_spectra"] == 4
            assert meta["n_fallback_excluded"] == 2
            assert meta["reference_strategy"] == sigaware_key

            # Subset must contain every configured strategy — the filter
            # is defined by signal_aware_fragment but applied to the
            # whole spectrum list so the comparison is apples-to-apples.
            for name in strategy_names:
                assert name in agg["per_strategy_annotation_driven"]
                assert (
                    agg["per_strategy_annotation_driven"][name]["n_spectra"] == 4
                )
                # Default view still sees all 6 records.
                assert agg["per_strategy"][name]["n_spectra"] == 6


class TestVisualizationsAndSave:
    def test_save_results_creates_files(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        analyser.aggregate_results([result])
        analyser.save_results()

        assert (analyser.output_dir / "masking_comparison_summary.json").exists()

    def test_generate_visualizations_no_crash(self, analyser):
        """Smoke test: visualisation code shouldn't crash."""
        all_results = []
        for _ in range(3):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
            )
            all_results.append(result)

        analyser.aggregate_results(all_results)
        # Should not raise
        analyser.generate_visualizations()
        assert (analyser.output_dir / "masking_training_signal_analysis.png").exists()
        assert (analyser.output_dir / "masking_prediction_difficulty.png").exists()

        # Per-strategy behavior figures should exist
        for name in analyser.strategies:
            assert (analyser.output_dir / name / "masking_behavior.png").exists()
            assert (analyser.output_dir / name / "group_size_tradeoff.png").exists()

    def test_print_summary_no_crash(self, analyser):
        """Smoke test: print_summary shouldn't crash."""
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        analyser.aggregate_results([result])
        analyser.print_summary()


class TestBehaviorHelpers:
    """Unit tests for static behavior metric helpers."""

    def test_run_lengths_basic(self):
        mask = np.array([True, True, False, True, True, True, False, True])
        runs = MaskingAnalyser._compute_run_lengths(mask)
        np.testing.assert_array_equal(runs, [2, 3, 1])

    def test_run_lengths_all_masked(self):
        mask = np.ones(5, dtype=bool)
        runs = MaskingAnalyser._compute_run_lengths(mask)
        np.testing.assert_array_equal(runs, [5])

    def test_run_lengths_none_masked(self):
        mask = np.zeros(5, dtype=bool)
        runs = MaskingAnalyser._compute_run_lengths(mask)
        assert len(runs) == 0

    def test_run_lengths_empty(self):
        mask = np.array([], dtype=bool)
        runs = MaskingAnalyser._compute_run_lengths(mask)
        assert len(runs) == 0

    def test_intensity_mask_curve_shape(self):
        intensity = np.random.uniform(0, 1, 100)
        mask = np.random.choice([True, False], 100)
        curve = MaskingAnalyser._compute_intensity_mask_curve(intensity, mask, n_bins=10)
        assert curve is not None
        assert len(curve) == 10
        # Each bin rate should be between 0 and 1
        assert np.all((curve >= 0) & (curve <= 1))

    def test_intensity_mask_curve_too_few_peaks(self):
        intensity = np.array([0.5, 0.6])
        mask = np.array([True, False])
        curve = MaskingAnalyser._compute_intensity_mask_curve(intensity, mask, n_bins=10)
        assert curve is None

    def test_spatial_mask_curve_shape(self):
        mz = np.linspace(100, 2000, 50)
        mask = np.random.choice([True, False], 50)
        edges = np.linspace(50, 2500, 26)
        curve = MaskingAnalyser._compute_spatial_mask_curve(mz, mask, edges)
        assert curve is not None
        assert len(curve) == 25


class TestStaticMethodRefactor:
    """Verify _analyze_masking_effect works as a static method."""

    def test_static_method_callable(self):
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 10,
            "n_unannotated_peaks": 20,
            "annotated_mask": [True] * 10 + [False] * 20,
            "annotation_labels": ["b1+"] * 10 + ["unannotated"] * 20,
        }
        valid_intensity = np.random.uniform(0.0, 1.0, size=30)
        mlm_mask = np.array([True] * 5 + [False] * 5 + [True] * 10 + [False] * 10)
        matched_annotations = ["b1+"] * 10 + [""] * 20
        feature_types = ["base"] * 10 + [None] * 20
        parent_annotations = [None] * 30

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )
        assert "annotated_mask_ratio" in result
        assert "unannotated_mask_ratio" in result
        assert result["overall_mask_ratio"] == pytest.approx(15.0 / 30.0)


class TestInformationLeakage:
    """Test _compute_information_leakage with synthetic fragment groups."""

    def test_no_leakage_when_all_children_masked(self):
        """Base masked + all children masked → leakage = 0."""
        # 4 peaks: base(masked), loss(masked), isotope(masked), unannotated(unmasked)
        mlm_mask = np.array([True, True, True, False])
        feature_types = ["base", "loss", "isotope", None]
        parent_annotations = [None, "b3+", "b3+", None]
        matched_annotations = ["b3+", "b3-H2O+", "b3+[+1]", ""]

        result = MaskingAnalyser._compute_information_leakage(
            mlm_mask, feature_types, parent_annotations, matched_annotations
        )
        assert result["n_masked_base"] == 1
        assert result["n_masked_base_with_leakage"] == 0
        assert result["leakage_ratio"] == 0.0

    def test_leakage_when_isotope_unmasked(self):
        """Base masked + isotope unmasked → leakage = 1."""
        mlm_mask = np.array([True, True, False])
        feature_types = ["base", "loss", "isotope"]
        parent_annotations = [None, "b3+", "b3+"]
        matched_annotations = ["b3+", "b3-H2O+", "b3+[+1]"]

        result = MaskingAnalyser._compute_information_leakage(
            mlm_mask, feature_types, parent_annotations, matched_annotations
        )
        assert result["n_masked_base"] == 1
        assert result["n_masked_base_with_leakage"] == 1
        assert result["leakage_ratio"] == 1.0
        assert result["leakage_by_type"]["isotope"] == 1
        assert result["leakage_by_type"]["loss"] == 0

    def test_leakage_when_loss_unmasked(self):
        """Base masked + loss unmasked → leakage via loss."""
        mlm_mask = np.array([True, False, True])
        feature_types = ["base", "loss", "isotope"]
        parent_annotations = [None, "y5+", "y5+"]
        matched_annotations = ["y5+", "y5-H2O+", "y5+[+1]"]

        result = MaskingAnalyser._compute_information_leakage(
            mlm_mask, feature_types, parent_annotations, matched_annotations
        )
        assert result["n_masked_base"] == 1
        assert result["leakage_ratio"] == 1.0
        assert result["leakage_by_type"]["loss"] == 1

    def test_no_masked_base_yields_zero(self):
        """No base ions masked → leakage_ratio = 0 (division guarded)."""
        mlm_mask = np.array([False, True, True])
        feature_types = ["base", "loss", "isotope"]
        parent_annotations = [None, "b3+", "b3+"]
        matched_annotations = ["b3+", "b3-H2O+", "b3+[+1]"]

        result = MaskingAnalyser._compute_information_leakage(
            mlm_mask, feature_types, parent_annotations, matched_annotations
        )
        assert result["n_masked_base"] == 0
        assert result["leakage_ratio"] == 0.0

    def test_multiple_groups(self):
        """Two base ions — one leaks, one doesn't."""
        mlm_mask = np.array([True, False, True, True])
        feature_types = ["base", "isotope", "base", "isotope"]
        parent_annotations = [None, "b3+", None, "y5+"]
        matched_annotations = ["b3+", "b3+[+1]", "y5+", "y5+[+1]"]

        result = MaskingAnalyser._compute_information_leakage(
            mlm_mask, feature_types, parent_annotations, matched_annotations
        )
        assert result["n_masked_base"] == 2
        assert result["n_masked_base_with_leakage"] == 1  # b3+ leaks
        assert result["leakage_ratio"] == pytest.approx(0.5)


class TestMaskArrayStorage:
    """Verify analyze_spectrum stores mask arrays and viz data."""

    def test_mlm_mask_valid_in_repeat(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        for name in analyser.strategies:
            rep = result[name]["repeats"][0]
            assert "mlm_mask_valid" in rep
            assert isinstance(rep["mlm_mask_valid"], np.ndarray)
            assert rep["mlm_mask_valid"].dtype == bool

    def test_viz_data_present(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
        )
        assert "_viz_data" in result
        viz = result["_viz_data"]
        assert "valid_mz" in viz
        assert "valid_intensity" in viz
        assert "n_valid" in viz
        assert viz["n_valid"] == 40  # 50 - 10 padding

    def test_viz_data_with_annotation(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        n_valid = 40
        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 10,
            "n_unannotated_peaks": n_valid - 10,
            "annotated_mask": [True] * 10 + [False] * (n_valid - 10),
            "annotation_labels": ["b1+"] * 10 + ["unannotated"] * (n_valid - 10),
        }
        matched_annotations = ["b1+"] * 10 + [""] * (n_valid - 10)
        feature_types = ["base"] * 10 + [None] * (n_valid - 10)
        parent_annotations = [None] * n_valid

        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
            theoretical_analysis=theo,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )
        viz = result["_viz_data"]
        assert "annotated_mask" in viz
        assert "theo_annotations" in viz
        assert "feature_types" in viz
        assert "parent_annotations" in viz


class TestLeakageInAggregation:
    """Verify leakage metrics flow through aggregation."""

    def test_leakage_in_aggregated_results(self, analyser):
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        n_valid = 40
        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 10,
            "n_unannotated_peaks": n_valid - 10,
            "annotated_mask": [True] * 10 + [False] * (n_valid - 10),
            "annotation_labels": ["b1+"] * 10 + ["unannotated"] * (n_valid - 10),
        }
        # Create parent-child relationship: 5 base ions, each with 1 isotope
        matched_annotations = []
        feature_types = []
        parent_annotations = []
        for i in range(5):
            matched_annotations.extend([f"b{i+1}+", f"b{i+1}+[+1]"])
            feature_types.extend(["base", "isotope"])
            parent_annotations.extend([None, f"b{i+1}+"])
        # Fill remaining with unannotated
        remaining = n_valid - 10
        matched_annotations.extend([""] * remaining)
        feature_types.extend([None] * remaining)
        parent_annotations.extend([None] * remaining)

        result = analyser.analyze_spectrum(
            spectra=spectra,
            spectra_mask=spectra_mask,
            precursor_charges=torch.tensor([2]),
            valid_mz=valid_mz,
            valid_intensity=valid_intensity,
            metadata={"frag_type": "HCD"},
            theoretical_analysis=theo,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )

        # Check leakage in repeats
        for name in analyser.strategies:
            rep = result[name]["repeats"][0]
            assert "leakage" in rep
            assert "leakage_ratio" in rep["leakage"]

        # Aggregate and check
        agg = analyser.aggregate_results([result])
        for name in analyser.strategies:
            assert "leakage" in agg["per_strategy"][name]

        # Check training_signal comparison table
        assert "training_signal" in agg["comparison"]


class TestIndividualSpectrumVisualization:
    """Smoke test: individual viz methods shouldn't crash."""

    def test_generate_individual_viz_no_crash(self, analyser):
        all_results = []
        for _ in range(3):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            n_valid = 40
            theo = {
                "sequence_available": True,
                "n_annotated_peaks": 10,
                "n_unannotated_peaks": n_valid - 10,
                "annotated_mask": [True] * 10 + [False] * (n_valid - 10),
                "annotation_labels": ["b1+"] * 10 + ["unannotated"] * (n_valid - 10),
            }
            matched_annotations = ["b1+"] * 10 + [""] * (n_valid - 10)
            feature_types = ["base"] * 10 + [None] * (n_valid - 10)
            parent_annotations = [None] * n_valid

            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD", "clean_sequence": "PEPTIDEK"},
                theoretical_analysis=theo,
                matched_annotations=matched_annotations,
                feature_types=feature_types,
                parent_annotations=parent_annotations,
            )
            all_results.append(result)

        analyser.aggregate_results(all_results)
        # Should not raise
        analyser.generate_individual_spectrum_visualizations(all_results, max_viz=2)

        # Check comparison figures exist
        comp_dir = analyser.output_dir / "individual_spectra_masking_strategy_comparison"
        assert comp_dir.exists()
        comp_files = list(comp_dir.glob("spectrum_*.png"))
        assert len(comp_files) == 2

        # Check per-strategy figures exist
        for name in analyser.strategies:
            strat_dir = analyser.output_dir / name / "individual_spectra"
            assert strat_dir.exists()
            strat_files = list(strat_dir.glob("spectrum_*.png"))
            assert len(strat_files) == 2


class TestPerTypeIntensityTracking:
    """Test per-type intensity tracking in _analyze_masking_effect."""

    def test_intensity_keys_present(self):
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 6,
            "n_unannotated_peaks": 4,
            "annotated_mask": [True] * 6 + [False] * 4,
            "annotation_labels": [
                "b-ion", "b-ion", "y-ion", "y-loss", "b-loss", "b-isotope",
                "unannotated", "unannotated", "unannotated", "unannotated",
            ],
        }
        valid_intensity = np.array([0.8, 0.6, 0.9, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2])
        mlm_mask = np.array([True, False, True, True, False, False, True, False, True, False])
        matched_annotations = ["b3+", "b4+", "y5+", "y5-H2O+", "b3-NH3+", "b3+[+1]", "", "", "", ""]
        feature_types = ["base", "base", "base", "loss", "loss", "isotope", None, None, None, None]
        parent_annotations = [None, None, None, "y5+", "b3+", "b3+", None, None, None, None]

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )

        # New intensity keys must be present
        assert "annotated_type_intensity_total" in result
        assert "annotated_type_intensity_masked" in result
        assert "annotated_type_intensity_mask_ratio" in result

        # Check that intensity totals are accumulated correctly
        int_total = result["annotated_type_intensity_total"]
        assert "b-ion" in int_total
        assert "y-ion" in int_total
        assert int_total["b-ion"] == pytest.approx(0.8 + 0.6)
        assert int_total["y-ion"] == pytest.approx(0.9)

        # Check masked intensity
        int_masked = result["annotated_type_intensity_masked"]
        assert int_masked.get("b-ion", 0.0) == pytest.approx(0.8)  # only first b-ion is masked
        assert int_masked.get("y-ion", 0.0) == pytest.approx(0.9)  # y-ion is masked
        assert int_masked.get("y-loss", 0.0) == pytest.approx(0.3)  # y-loss is masked

        # Check mask ratio
        int_ratio = result["annotated_type_intensity_mask_ratio"]
        assert int_ratio["b-ion"] == pytest.approx(0.8 / 1.4)
        assert int_ratio["y-ion"] == pytest.approx(1.0)

    def test_intensity_aggregation(self, analyser):
        """Verify per-type intensity flows through aggregation."""
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        n_valid = 40
        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 12,
            "n_unannotated_peaks": n_valid - 12,
            "annotated_mask": [True] * 12 + [False] * (n_valid - 12),
            "annotation_labels": (
                ["b-ion"] * 4 + ["y-ion"] * 4 + ["b-loss"] * 2 + ["b-isotope"] * 2
                + ["unannotated"] * (n_valid - 12)
            ),
        }
        matched_annotations = (
            ["b1+", "b2+", "b3+", "b4+", "y1+", "y2+", "y3+", "y4+",
             "b1-H2O+", "b2-H2O+", "b1+[+1]", "b2+[+1]"]
            + [""] * (n_valid - 12)
        )
        feature_types = (
            ["base"] * 4 + ["base"] * 4 + ["loss"] * 2 + ["isotope"] * 2
            + [None] * (n_valid - 12)
        )
        parent_annotations = (
            [None] * 4 + [None] * 4 + ["b1+", "b2+", "b1+", "b2+"]
            + [None] * (n_valid - 12)
        )

        all_results = []
        for _ in range(3):
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
                theoretical_analysis=theo,
                matched_annotations=matched_annotations,
                feature_types=feature_types,
                parent_annotations=parent_annotations,
            )
            all_results.append(result)

        agg = analyser.aggregate_results(all_results)

        for name in analyser.strategies:
            eff = agg["per_strategy"][name].get("masking_effect", {})
            # Intensity keys must be present in aggregated results
            assert "annotated_type_intensity_total" in eff
            assert "annotated_type_intensity_masked" in eff
            assert "annotated_type_intensity_mask_ratio" in eff


class TestCustomIonMetrics:
    """Test custom ion metrics in _analyze_masking_effect."""

    def test_custom_ion_keys_present(self):
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 5,
            "n_unannotated_peaks": 5,
            "annotated_mask": [True] * 5 + [False] * 5,
            "annotation_labels": ["b-ion"] * 5 + ["unannotated"] * 5,
        }
        valid_intensity = np.array([0.8, 0.6, 0.9, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, 0.1])
        mlm_mask = np.array([True, False, True, False, True, True, True, False, False, True])
        matched_annotations = ["b1+", "b2+", "b3+", "b4+", "b5+", "", "", "", "", ""]
        feature_types = ["base"] * 5 + [None] * 5
        parent_annotations = [None] * 10

        # 3 custom ion peaks at indices 5, 6, 9 (unannotated)
        custom_ion_peak_mask = np.array([False] * 5 + [True, True, False, False, True])

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
            custom_ion_peak_mask=custom_ion_peak_mask,
        )

        # Custom ion keys must be present
        assert "custom_n_peaks" in result
        assert "custom_n_masked" in result
        assert "custom_mask_ratio" in result
        assert "custom_mask_budget_fraction" in result
        assert "custom_intensity_budget_fraction" in result

        # Verify counts
        assert result["custom_n_peaks"] == 3  # indices 5, 6, 9
        # Masked custom: indices 5 (masked), 6 (masked), 9 (masked) = 3
        assert result["custom_n_masked"] == 3
        assert result["custom_mask_ratio"] == pytest.approx(1.0)

    def test_custom_ion_no_mask(self):
        """Test custom ion metrics when no custom_ion_peak_mask is provided."""
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 3,
            "n_unannotated_peaks": 2,
            "annotated_mask": [True] * 3 + [False] * 2,
            "annotation_labels": ["b-ion"] * 3 + ["unannotated"] * 2,
        }
        valid_intensity = np.array([0.8, 0.6, 0.9, 0.3, 0.2])
        mlm_mask = np.array([True, False, True, True, False])
        matched_annotations = ["b1+", "b2+", "b3+", "", ""]
        feature_types = ["base"] * 3 + [None] * 2
        parent_annotations = [None] * 5

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )

        assert result["custom_n_peaks"] == 0
        assert result["custom_n_masked"] == 0
        assert result["custom_mask_budget_fraction"] == 0.0

    def test_custom_ion_in_aggregation(self, analyser):
        """Verify custom ion metrics aggregate correctly."""
        spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
        n_valid = 40
        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 10,
            "n_unannotated_peaks": n_valid - 10,
            "annotated_mask": [True] * 10 + [False] * (n_valid - 10),
            "annotation_labels": ["b-ion"] * 10 + ["unannotated"] * (n_valid - 10),
        }
        matched_annotations = ["b1+"] * 10 + [""] * (n_valid - 10)
        feature_types = ["base"] * 10 + [None] * (n_valid - 10)
        parent_annotations = [None] * n_valid
        custom_mask = np.zeros(n_valid, dtype=bool)
        custom_mask[15] = True
        custom_mask[20] = True

        all_results = []
        for _ in range(3):
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
                theoretical_analysis=theo,
                matched_annotations=matched_annotations,
                feature_types=feature_types,
                parent_annotations=parent_annotations,
                custom_ion_peak_mask=custom_mask,
            )
            all_results.append(result)

        agg = analyser.aggregate_results(all_results)

        for name in analyser.strategies:
            eff = agg["per_strategy"][name].get("masking_effect", {})
            assert "avg_custom_mask_budget_fraction" in eff
            assert "avg_custom_intensity_budget_fraction" in eff
            assert "total_custom_n_peaks" in eff


class TestNewFigures:
    """Smoke tests for new summary and fragment ion figures."""

    def _make_annotated_results(self, analyser, n_spectra=5):
        """Helper to create results with full annotation data."""
        all_results = []
        for _ in range(n_spectra):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            n_valid = 40
            theo = {
                "sequence_available": True,
                "n_annotated_peaks": 12,
                "n_unannotated_peaks": n_valid - 12,
                "annotated_mask": [True] * 12 + [False] * (n_valid - 12),
                "annotation_labels": (
                    ["b-ion"] * 4 + ["y-ion"] * 4 + ["b-loss"] * 2 + ["b-isotope"] * 2
                    + ["unannotated"] * (n_valid - 12)
                ),
            }
            matched_annotations = (
                ["b1+", "b2+", "b3+", "b4+", "y1+", "y2+", "y3+", "y4+",
                 "b1-H2O+", "b2-H2O+", "b1+[+1]", "b2+[+1]"]
                + [""] * (n_valid - 12)
            )
            feature_types = (
                ["base"] * 4 + ["base"] * 4 + ["loss"] * 2 + ["isotope"] * 2
                + [None] * (n_valid - 12)
            )
            parent_annotations = (
                [None] * 4 + [None] * 4 + ["b1+", "b2+", "b1+", "b2+"]
                + [None] * (n_valid - 12)
            )

            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
                theoretical_analysis=theo,
                matched_annotations=matched_annotations,
                feature_types=feature_types,
                parent_annotations=parent_annotations,
            )
            all_results.append(result)
        return all_results

    def test_masking_summary_figure(self, analyser):
        """Smoke test: masking_summary.png is created without error."""
        all_results = self._make_annotated_results(analyser)
        analyser.aggregate_results(all_results)
        analyser.generate_visualizations()
        assert (analyser.output_dir / "masking_summary.png").exists()

    def test_fragment_ion_analysis_figure(self, analyser):
        """Smoke test: masking_fragment_ion_analysis.png is created without error."""
        all_results = self._make_annotated_results(analyser)
        analyser.aggregate_results(all_results)
        analyser.generate_visualizations()
        assert (analyser.output_dir / "masking_fragment_ion_analysis.png").exists()

    def test_existing_figures_still_created(self, analyser):
        """Verify existing figures are still created alongside new ones."""
        all_results = self._make_annotated_results(analyser)
        analyser.aggregate_results(all_results)
        analyser.generate_visualizations()

        # New figures
        assert (analyser.output_dir / "masking_summary.png").exists()
        assert (analyser.output_dir / "masking_fragment_ion_analysis.png").exists()
        # Existing figures
        assert (analyser.output_dir / "masking_training_signal_analysis.png").exists()
        assert (analyser.output_dir / "masking_prediction_difficulty.png").exists()

    def test_custom_ion_columns_in_comparison_table(self, analyser):
        """Verify custom ion columns in training_signal comparison table."""
        all_results = self._make_annotated_results(analyser, n_spectra=3)
        agg = analyser.aggregate_results(all_results)
        signal_df = agg["comparison"]["training_signal"]
        assert "custom_budget_frac" in signal_df.columns
        assert "custom_int_budget_frac" in signal_df.columns


class TestFragmentGroupStatus:
    """Tests for the new fragment group masking status figure and supporting data."""

    def _make_annotated_results(self, analyser, n_spectra=5):
        """Helper that creates results with fragment groups (base + isotope)."""
        all_results = []
        for _ in range(n_spectra):
            spectra, spectra_mask, valid_mz, valid_intensity = _make_synthetic_spectrum()
            n_valid = 40
            theo = {
                "sequence_available": True,
                "n_annotated_peaks": 12,
                "n_unannotated_peaks": n_valid - 12,
                "annotated_mask": [True] * 12 + [False] * (n_valid - 12),
                "annotation_labels": (
                    ["b-ion"] * 4 + ["y-ion"] * 4 + ["b-loss"] * 2 + ["b-isotope"] * 2
                    + ["unannotated"] * (n_valid - 12)
                ),
            }
            matched_annotations = (
                ["b1+", "b2+", "b3+", "b4+", "y1+", "y2+", "y3+", "y4+",
                 "b1-H2O+", "b2-H2O+", "b1+[+1]", "b2+[+1]"]
                + [""] * (n_valid - 12)
            )
            feature_types = (
                ["base"] * 4 + ["base"] * 4 + ["loss"] * 2 + ["isotope"] * 2
                + [None] * (n_valid - 12)
            )
            parent_annotations = (
                [None] * 4 + [None] * 4 + ["b1+", "b2+", "b1+", "b2+"]
                + [None] * (n_valid - 12)
            )
            result = analyser.analyze_spectrum(
                spectra=spectra,
                spectra_mask=spectra_mask,
                precursor_charges=torch.tensor([2]),
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                metadata={"frag_type": "HCD"},
                theoretical_analysis=theo,
                matched_annotations=matched_annotations,
                feature_types=feature_types,
                parent_annotations=parent_annotations,
            )
            all_results.append(result)
        return all_results

    def test_group_mask_fractions_in_per_spectrum(self):
        """Verify group_mask_fractions is returned from _analyze_masking_effect."""
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 6,
            "n_unannotated_peaks": 4,
            "annotated_mask": [True] * 6 + [False] * 4,
            "annotation_labels": [
                "b-ion", "b-ion", "y-ion", "y-ion", "b-isotope", "b-isotope",
                "unannotated", "unannotated", "unannotated", "unannotated",
            ],
        }
        valid_intensity = np.array([0.8, 0.6, 0.9, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2])
        mlm_mask = np.array([True, True, True, False, True, False, True, False, True, False])
        matched_annotations = ["b1+", "b2+", "y1+", "y2+", "b1+[+1]", "b2+[+1]", "", "", "", ""]
        feature_types = ["base", "base", "base", "base", "isotope", "isotope", None, None, None, None]
        parent_annotations = [None, None, None, None, "b1+", "b2+", None, None, None, None]

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )

        assert "group_mask_fractions" in result
        fracs = result["group_mask_fractions"]
        # Groups: b1+ (base + isotope), b2+ (base + isotope), y1+ (base), y2+ (base)
        # b1+: base masked, isotope masked -> 2/2 = 1.0
        # b2+: base masked, isotope unmasked -> 1/2 = 0.5
        # y1+: base masked -> 1/1 = 1.0
        # y2+: base unmasked -> 0/1 = 0.0
        assert len(fracs) == 4
        assert sorted(fracs) == pytest.approx([0.0, 0.5, 1.0, 1.0])

    def test_series_partially_unmasked_in_per_spectrum(self):
        """Verify per-series partially_masked and unmasked are returned."""
        from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser

        theo = {
            "sequence_available": True,
            "n_annotated_peaks": 6,
            "n_unannotated_peaks": 4,
            "annotated_mask": [True] * 6 + [False] * 4,
            "annotation_labels": [
                "b-ion", "b-ion", "y-ion", "y-ion", "b-isotope", "b-isotope",
                "unannotated", "unannotated", "unannotated", "unannotated",
            ],
        }
        valid_intensity = np.array([0.8, 0.6, 0.9, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2])
        mlm_mask = np.array([True, True, True, False, True, False, True, False, True, False])
        matched_annotations = ["b1+", "b2+", "y1+", "y2+", "b1+[+1]", "b2+[+1]", "", "", "", ""]
        feature_types = ["base", "base", "base", "base", "isotope", "isotope", None, None, None, None]
        parent_annotations = [None, None, None, None, "b1+", "b2+", None, None, None, None]

        result = TheoreticalAnalyser._analyze_masking_effect(
            theoretical_analysis=theo,
            valid_intensity=valid_intensity,
            mlm_mask=mlm_mask,
            matched_annotations=matched_annotations,
            feature_types=feature_types,
            parent_annotations=parent_annotations,
        )

        assert "series_group_partially_masked" in result
        assert "series_group_unmasked" in result
        # b-series: b1+ fully masked, b2+ partially masked (1/2) -> 1 partially
        # y-series: y1+ fully masked, y2+ unmasked -> 1 unmasked
        assert result["series_group_partially_masked"].get("b", 0) == 1
        assert result["series_group_unmasked"].get("y", 0) == 1

    def test_distributional_data_in_aggregation(self, analyser):
        """Verify per-spectrum distributional data is collected in aggregation."""
        all_results = self._make_annotated_results(analyser)
        agg = analyser.aggregate_results(all_results)

        for name in analyser.strategies:
            eff = agg["per_strategy"][name].get("masking_effect", {})
            # Per-spectrum distributions
            assert "dist_n_fully_masked" in eff
            assert "dist_n_partially_masked" in eff
            assert "dist_n_unmasked" in eff
            # Should have one entry per spectrum × repeats
            assert len(eff["dist_n_fully_masked"]) > 0

            # Per-group partial fractions
            assert "dist_partial_group_fractions" in eff

            # Per-series breakdown should include partially/unmasked
            assert "series_group_partially_masked" in eff
            assert "series_group_unmasked" in eff

    def test_fragment_group_status_figure(self, analyser):
        """Smoke test: masking_fragment_group_status.png is created without error."""
        all_results = self._make_annotated_results(analyser)
        analyser.aggregate_results(all_results)
        analyser.generate_visualizations()
        assert (analyser.output_dir / "masking_fragment_group_status.png").exists()

    def test_three_way_breakdown_sums_to_total(self, analyser):
        """The 3-way group breakdown must sum to total for each strategy."""
        all_results = self._make_annotated_results(analyser, n_spectra=10)
        agg = analyser.aggregate_results(all_results)

        for name in analyser.strategies:
            eff = agg["per_strategy"][name].get("masking_effect", {})
            total = eff.get("total_n_fragment_groups_total", 0)
            full = eff.get("total_n_fragment_groups_fully_masked", 0)
            partial = eff.get("total_n_fragment_groups_partially_masked", 0)
            unmask = eff.get("total_n_fragment_groups_unmasked", 0)
            if total > 0:
                assert full + partial + unmask == total, (
                    f"{name}: {full}+{partial}+{unmask} != {total}"
                )
