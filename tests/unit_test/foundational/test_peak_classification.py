"""Tests for instanovo_fm.utils.peak_classification."""
from __future__ import annotations

import pytest

from instanovo_fm.utils.peak_classification import (
    classify_peaks_batch,
    compute_spectrum_quality,
    extract_charge_from_annotation,
    extract_fragment_position,
    extract_ion_type,
    group_fragment_key,
    parse_peak_label,
)


# ---------------------------------------------------------------------------
# parse_peak_label
# ---------------------------------------------------------------------------
class TestParsePeakLabel:
    """Tests for parse_peak_label()."""

    def test_unannotated_none_feature_type(self):
        assert parse_peak_label(None, None, None) == "unannotated"

    def test_unannotated_none_with_annotation(self):
        # feature_type is None even though annotation is present
        assert parse_peak_label(None, "b3+", None) == "unannotated"

    # --- b / y ions ---
    def test_b_ion(self):
        assert parse_peak_label("base", "b3+", None) == "b-ion"

    def test_y_ion(self):
        assert parse_peak_label("base", "y5++", None) == "y-ion"

    def test_b_loss(self):
        assert parse_peak_label("loss", "b3+-H2O", "b3+") == "b-loss"

    def test_y_loss(self):
        assert parse_peak_label("loss", "y5+-NH3", "y5+") == "y-loss"

    def test_b_isotope(self):
        assert parse_peak_label("isotope", "b3+[+1]", "b3+") == "b-isotope"

    def test_y_isotope(self):
        assert parse_peak_label("isotope", "y5+[+1]", "y5+") == "y-isotope"

    # --- precursor ---
    def test_precursor_base(self):
        assert parse_peak_label("precursor", "p^2", None) == "precursor"

    def test_precursor_isotope(self):
        assert parse_peak_label("isotope", "p^2[+1]", "p^2") == "precursor-isotope"

    def test_precursor_with_p_dash(self):
        assert parse_peak_label("base", "p-H2O", None) == "precursor"

    # --- custom ---
    def test_custom(self):
        assert parse_peak_label("custom", "custom:immonium_His@110.0713", None) == "custom"

    def test_custom_no_annotation(self):
        assert parse_peak_label("custom", None, None) == "custom"

    # --- a-ions ---
    def test_a_ion(self):
        assert parse_peak_label("base", "a2+", None) == "a-ion"

    def test_a_loss(self):
        assert parse_peak_label("loss", "a2+-H2O", "a2+") == "a-loss"

    def test_a_isotope(self):
        assert parse_peak_label("isotope", "a2+[+1]", "a2+") == "a-isotope"

    # --- edge cases ---
    def test_other_empty_annotation(self):
        assert parse_peak_label("base", "", None) == "other"

    def test_other_non_string_annotation(self):
        assert parse_peak_label("base", 123, None) == "other"


# ---------------------------------------------------------------------------
# group_fragment_key
# ---------------------------------------------------------------------------
class TestGroupFragmentKey:
    """Tests for group_fragment_key()."""

    def test_base_ion_returns_annotation(self):
        assert group_fragment_key("base", "b3+", None) == "b3+"

    def test_precursor_returns_annotation(self):
        assert group_fragment_key("precursor", "p^2", None) == "p^2"

    def test_isotope_returns_parent(self):
        assert group_fragment_key("isotope", "b3+[+1]", "b3+") == "b3+"

    def test_isotope_strips_suffix_when_no_parent(self):
        assert group_fragment_key("isotope", "y5+[+2]", None) == "y5+"

    def test_loss_returns_none(self):
        assert group_fragment_key("loss", "b3+-H2O", "b3+") is None

    def test_unannotated_returns_none(self):
        assert group_fragment_key(None, None, None) is None

    def test_custom_returns_annotation(self):
        ann = "custom:immonium_His@110.0713"
        assert group_fragment_key("custom", ann, None) == ann

    def test_custom_empty_annotation_returns_none(self):
        assert group_fragment_key("custom", "", None) is None

    def test_custom_none_annotation_returns_none(self):
        assert group_fragment_key("custom", None, None) is None

    # Inferred type from annotation string (feature_type=None)
    def test_inferred_precursor(self):
        assert group_fragment_key(None, "p^3", None) == "p^3"

    def test_inferred_isotope(self):
        assert group_fragment_key(None, "b3+[+1]", None) == "b3+"

    def test_inferred_loss(self):
        assert group_fragment_key(None, "b3+-H2O", None) is None

    def test_inferred_base(self):
        assert group_fragment_key(None, "b3+", None) == "b3+"


# ---------------------------------------------------------------------------
# classify_peaks_batch
# ---------------------------------------------------------------------------
class TestClassifyPeaksBatch:
    """Tests for classify_peaks_batch()."""

    def test_full_spectrum(self):
        feature_types = [
            "base", "base", "loss", "isotope",
            "precursor", "isotope", "custom", None,
        ]
        annotations = [
            "b3+", "y5++", "b3+-H2O", "y5+[+1]",
            "p^2", "p^2[+1]", "custom:immonium_His@110.0713", None,
        ]
        parents = [
            None, None, "b3+", "y5+",
            None, "p^2", None, None,
        ]

        detail_labels, frag_keys = classify_peaks_batch(
            feature_types, annotations, parents,
        )

        assert len(detail_labels) == 8
        assert len(frag_keys) == 8

        assert detail_labels[0] == "b-ion"
        assert detail_labels[1] == "y-ion"
        assert detail_labels[2] == "b-loss"
        assert detail_labels[3] == "y-isotope"
        assert detail_labels[4] == "precursor"
        assert detail_labels[5] == "precursor-isotope"
        assert detail_labels[6] == "custom"
        assert detail_labels[7] == "unannotated"

        assert frag_keys[0] == "b3+"
        assert frag_keys[1] == "y5++"
        assert frag_keys[2] is None  # loss
        assert frag_keys[3] == "y5+"  # isotope -> parent
        assert frag_keys[4] == "p^2"
        assert frag_keys[5] == "p^2"  # precursor-isotope -> parent
        assert frag_keys[6] == "custom:immonium_His@110.0713"
        assert frag_keys[7] is None  # unannotated

    def test_empty_lists(self):
        detail_labels, frag_keys = classify_peaks_batch([], [], [])
        assert detail_labels == []
        assert frag_keys == []

    def test_all_unannotated(self):
        detail_labels, frag_keys = classify_peaks_batch(
            [None, None, None],
            [None, None, None],
            [None, None, None],
        )
        assert detail_labels == ["unannotated", "unannotated", "unannotated"]
        assert frag_keys == [None, None, None]


# ---------------------------------------------------------------------------
# extract_ion_type
# ---------------------------------------------------------------------------
class TestExtractIonType:
    """Tests for extract_ion_type()."""

    def test_b_ion(self):
        assert extract_ion_type("b3+") == "b"

    def test_y_ion(self):
        assert extract_ion_type("y5+-NH3") == "y"

    def test_a_ion(self):
        assert extract_ion_type("a2++") == "a"

    def test_precursor(self):
        assert extract_ion_type("p^2") == "p"

    def test_none(self):
        assert extract_ion_type(None) == "unknown"

    def test_empty_string(self):
        assert extract_ion_type("") == "unknown"

    def test_non_string(self):
        assert extract_ion_type(42) == "unknown"

    def test_isotope_annotation(self):
        assert extract_ion_type("b3+[+1]") == "b"


# ---------------------------------------------------------------------------
# extract_fragment_position
# ---------------------------------------------------------------------------
class TestExtractFragmentPosition:
    """Tests for extract_fragment_position()."""

    def test_b3(self):
        assert extract_fragment_position("b3+") == 3

    def test_y12(self):
        assert extract_fragment_position("y12++") == 12

    def test_a2(self):
        assert extract_fragment_position("a2+") == 2

    def test_precursor_returns_minus1(self):
        assert extract_fragment_position("p^2") == -1

    def test_precursor_loss_returns_minus1(self):
        assert extract_fragment_position("p-H2O^2") == -1

    def test_none(self):
        assert extract_fragment_position(None) == -1

    def test_empty(self):
        assert extract_fragment_position("") == -1

    def test_isotope_annotation(self):
        assert extract_fragment_position("b7+[+1]") == 7


# ---------------------------------------------------------------------------
# extract_charge_from_annotation
# ---------------------------------------------------------------------------
class TestExtractChargeFromAnnotation:
    """Tests for extract_charge_from_annotation()."""

    def test_single_charge(self):
        assert extract_charge_from_annotation("b3+") == 1

    def test_double_charge(self):
        assert extract_charge_from_annotation("y5++") == 2

    def test_triple_charge(self):
        assert extract_charge_from_annotation("b3+++") == 3

    def test_precursor_caret_notation(self):
        assert extract_charge_from_annotation("p^2") == 2

    def test_precursor_loss_caret(self):
        assert extract_charge_from_annotation("p-H2O^3") == 3

    def test_none_returns_1(self):
        assert extract_charge_from_annotation(None) == 1

    def test_empty_returns_1(self):
        assert extract_charge_from_annotation("") == 1

    def test_isotope_bracket_stripped(self):
        # "b3+[+1]" → strip "[+1]" → "b3+" → 1
        assert extract_charge_from_annotation("b3+[+1]") == 1


# ---------------------------------------------------------------------------
# compute_spectrum_quality
# ---------------------------------------------------------------------------
class TestComputeSpectrumQuality:
    """Tests for compute_spectrum_quality()."""

    def test_simple_b_and_y_ions(self):
        # Peptide: ABCDE (seq_len=5, max_cleavage_sites=4, sites 1..4)
        # b1+, b2+, y1+, y2+ → sites 1, 2, 4, 3 → 4 cleavage sites
        feature_types = ["base", "base", "base", "base"]
        annotations = ["b1+", "b2+", "y1+", "y2+"]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=5)

        assert result["max_cleavage_sites"] == 4
        assert result["n_fragment_groups"] == 4
        # b1→site1, b2→site2, y1→site(5-1)=4, y2→site(5-2)=3
        assert result["n_cleavage_sites"] == 4
        assert result["backbone_coverage"] == pytest.approx(1.0)

    def test_overlapping_cleavage_sites(self):
        # b3+ maps to site 3, y2+ with seq_len=5 maps to site 5-2=3
        # Same cleavage site → only 1 unique site
        feature_types = ["base", "base"]
        annotations = ["b3+", "y2+"]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=5)

        assert result["n_fragment_groups"] == 2
        assert result["n_cleavage_sites"] == 1
        assert result["backbone_coverage"] == pytest.approx(0.25)

    def test_ignores_losses_and_isotopes(self):
        # Only "base" feature_type peaks should be counted
        feature_types = ["base", "loss", "isotope", "precursor", None]
        annotations = ["b3+", "b3+-H2O", "b3+[+1]", "p^2", None]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=10)

        assert result["n_fragment_groups"] == 1  # only (b, 3)
        assert result["n_cleavage_sites"] == 1  # site 3

    def test_empty_spectrum(self):
        result = compute_spectrum_quality([], [], seq_len=5)
        assert result["backbone_coverage"] == 0.0
        assert result["n_fragment_groups"] == 0
        assert result["n_cleavage_sites"] == 0

    def test_seq_len_1_no_sites(self):
        # Single amino acid → max_cleavage_sites = 0
        result = compute_spectrum_quality(["base"], ["b1+"], seq_len=1)
        assert result["max_cleavage_sites"] == 0
        assert result["backbone_coverage"] == 0.0

    def test_precursor_excluded_from_groups(self):
        # Precursor ions (p^) have position=-1, should be skipped
        feature_types = ["base", "base"]
        annotations = ["p^2", "b3+"]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=10)

        # p^2 → position=-1, skipped. Only b3 counted.
        assert result["n_fragment_groups"] == 1
        assert result["n_cleavage_sites"] == 1

    def test_c_terminal_boundary(self):
        # y-ion at position == seq_len would map to site=0, which should be excluded
        feature_types = ["base"]
        annotations = ["y5+"]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=5)

        # y5 with seq_len=5 → site = 5-5 = 0 → excluded (0 < site < seq_len)
        assert result["n_fragment_groups"] == 1
        assert result["n_cleavage_sites"] == 0
        assert result["backbone_coverage"] == 0.0

    def test_duplicate_groups_deduped(self):
        # Two b3+ peaks should only count as 1 group
        feature_types = ["base", "base"]
        annotations = ["b3+", "b3++"]
        result = compute_spectrum_quality(feature_types, annotations, seq_len=10)

        # Both are (b, 3) → 1 group, 1 cleavage site
        assert result["n_fragment_groups"] == 1
        assert result["n_cleavage_sites"] == 1
