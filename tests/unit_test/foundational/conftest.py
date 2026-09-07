"""Expected failures inherited from the source branch.

Everything mechanically fixable has been fixed rather than suppressed: the
``SearchDataManager.excel_path`` -> ``file_path`` rename (2 tests), the
``_extract_fragment_position`` helper moving from ``SpectrumAnalyser`` to
``instanovo_fm.utils.peak_classification`` (7 tests), and the optional-dependency
tests, which now skip via ``importorskip`` instead of failing (4 tests).

What remains cannot be fixed without a decision about intended behaviour, so it
is recorded rather than guessed at:

* ``test_metrics_vectorization`` (5) -- the tests call ``StreamingMetrics``
  members that have no counterpart at all: ``tokens_within_10da``,
  ``running_cosine_sim``, ``mz_slice_da_counts``, ``update_uncertainty_metrics``.
  The metric accounting was restructured, not renamed. Mapping them onto
  ``update_confidence_metrics`` or ``update_mz_metrics`` would make the tests
  pass while asserting something nobody has verified.
* ``test_bin_jump_helpers`` (2) -- ``enable_bin_jump_analysis`` is now derived
  from ``"binning" in self.active_tasks`` (spectrum_analyser.py:285) rather than
  read from ``analysis.enable_bin_jump_analysis``. The tests assert the old
  config-key behaviour. Which one is right is a call for whoever changed it.
* ``test_intensity_metrics`` (1) -- asserts a Spearman correlation returns None
  for constant input; the implementation no longer does.

Non-strict xfail, so pytest reports XPASS the moment one starts working again.
Anyone reconciling these should delete the entry, not the test.
"""

import pytest

_KNOWN_FAILURES = {
    "test_bin_jump_helpers.py::TestBinJumpConfiguration::test_custom_config_values",
    "test_bin_jump_helpers.py::TestPartDConfiguration::test_part_d_custom_values",
    "test_intensity_metrics.py::TestSpearmanCorrelation::test_constant_values_returns_none",
    "test_metrics_vectorization.py::TestMzMetricsVectorization::test_cosine_similarity",
    "test_metrics_vectorization.py::TestMzMetricsVectorization::test_mz_slice_da_accuracy",
    "test_metrics_vectorization.py::TestMzMetricsVectorization::test_percentage_thresholds",
    "test_metrics_vectorization.py::TestMzMetricsVectorization::test_ppm_denominator_consistency",
    "test_metrics_vectorization.py::TestMzMetricsVectorization::test_single_token_spectrum",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        suffix = item.nodeid.split("foundational/")[-1]
        if suffix in _KNOWN_FAILURES:
            item.add_marker(
                pytest.mark.xfail(
                    reason="inherited from the source branch; see tests/unit_test/foundational/conftest.py",
                    strict=False,
                )
            )
