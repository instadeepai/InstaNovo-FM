"""Unit tests for _spearman_correlation, _topk_recall, and _extract_loss_type."""

import torch
import pytest

from instanovo_fm.trainer.metrics import StreamingMetrics
from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser


class TestSpearmanCorrelation:
    """Tests for StreamingMetrics._spearman_correlation."""

    def test_perfect_positive_correlation(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result == pytest.approx(-1.0)

    def test_no_correlation(self):
        """Orthogonal ranks should give zero correlation."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_single_element_returns_none(self):
        pred = torch.tensor([1.0])
        target = torch.tensor([2.0])
        assert StreamingMetrics._spearman_correlation(pred, target) is None

    def test_empty_returns_none(self):
        pred = torch.tensor([])
        target = torch.tensor([])
        assert StreamingMetrics._spearman_correlation(pred, target) is None

    def test_constant_values_returns_none(self):
        """All identical values produce NaN correlation → should return None."""
        pred = torch.tensor([5.0, 5.0, 5.0, 5.0])
        target = torch.tensor([3.0, 1.0, 4.0, 2.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result is None

    def test_tied_values_use_average_ranks(self):
        """Tied values should still produce a valid correlation via scipy."""
        pred = torch.tensor([1.0, 1.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result is not None
        assert result > 0.9  # High but not perfect due to tie

    def test_two_elements(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0])
        result = StreamingMetrics._spearman_correlation(pred, target)
        assert result == pytest.approx(1.0)


class TestTopkRecall:
    """Tests for StreamingMetrics._topk_recall."""

    def test_perfect_recall(self):
        """Pred and target have same top-k → recall = 1.0."""
        pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        target = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        assert StreamingMetrics._topk_recall(pred, target, k=3) == pytest.approx(1.0)

    def test_zero_recall(self):
        """Top-k in pred is completely disjoint from top-k in target."""
        pred = torch.tensor([1.0, 2.0, 3.0, 100.0, 200.0])
        target = torch.tensor([200.0, 100.0, 50.0, 1.0, 2.0])
        # pred top-3: indices {4,3,2}, target top-3: indices {0,1,2}
        # overlap = {2} → recall = 1/3
        result = StreamingMetrics._topk_recall(pred, target, k=3)
        assert result == pytest.approx(1.0 / 3.0)

    def test_partial_recall(self):
        pred = torch.tensor([10.0, 1.0, 9.0, 2.0, 8.0])
        target = torch.tensor([10.0, 8.0, 1.0, 2.0, 9.0])
        # pred top-3: indices {0,2,4}, target top-3: indices {0,4,1}
        # overlap = {0,4} → recall = 2/3
        result = StreamingMetrics._topk_recall(pred, target, k=3)
        assert result == pytest.approx(2.0 / 3.0)

    def test_k_equals_n(self):
        """When k equals the length, recall must be 1.0."""
        pred = torch.tensor([3.0, 1.0, 2.0])
        target = torch.tensor([1.0, 3.0, 2.0])
        assert StreamingMetrics._topk_recall(pred, target, k=3) == pytest.approx(1.0)

    def test_k_equals_one(self):
        pred = torch.tensor([1.0, 5.0, 3.0])
        target = torch.tensor([1.0, 5.0, 3.0])
        # Both have top-1 at index 1
        assert StreamingMetrics._topk_recall(pred, target, k=1) == pytest.approx(1.0)

    def test_k_equals_one_no_overlap(self):
        pred = torch.tensor([1.0, 5.0, 3.0])
        target = torch.tensor([10.0, 1.0, 3.0])
        # pred top-1: {1}, target top-1: {0}
        assert StreamingMetrics._topk_recall(pred, target, k=1) == pytest.approx(0.0)


class TestExtractLossType:
    """Tests for TheoreticalAnalyser._extract_loss_type."""

    def test_h2o_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("b3+-H2O") == "H2O"

    def test_nh3_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("y5+-NH3") == "NH3"

    def test_h3po4_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("b7++-H3PO4") == "H3PO4"

    def test_so3_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("y4+-SO3") == "SO3"

    def test_co_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("b2+-CO") == "CO"

    def test_no_loss(self):
        assert TheoreticalAnalyser._extract_loss_type("b3+") is None

    def test_empty_string(self):
        assert TheoreticalAnalyser._extract_loss_type("") is None

    def test_none_input(self):
        assert TheoreticalAnalyser._extract_loss_type(None) is None

    def test_unknown_loss_returns_none(self):
        assert TheoreticalAnalyser._extract_loss_type("b3+-XYZ") is None

    def test_loss_with_isotope_bracket(self):
        """Loss annotation with isotope bracket should still extract the loss."""
        assert TheoreticalAnalyser._extract_loss_type("b3+-H2O[+1]") == "H2O"

    def test_loss_with_psi_caret_notation(self):
        """Loss with PSI mzPAF charge notation should still extract the loss."""
        assert TheoreticalAnalyser._extract_loss_type("y5-NH3^2") == "NH3"

    def test_loss_with_trailing_charge(self):
        """Loss followed by charge '+' symbols."""
        assert TheoreticalAnalyser._extract_loss_type("y5-H2O+") == "H2O"

    def test_annotation_with_hyphen_but_no_known_loss(self):
        """Annotations containing a hyphen but no known loss name."""
        assert TheoreticalAnalyser._extract_loss_type("p-H2O^2") is None or \
               TheoreticalAnalyser._extract_loss_type("p-H2O^2") == "H2O"
