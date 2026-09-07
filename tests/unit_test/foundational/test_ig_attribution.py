"""Regression tests for the IG attribution peak categorizer.

Covers the bug fixes landed with the category-system overhaul:
- Complementary-pair detection no longer adds a spurious H2O.
- Ladder-neighbor label uses series indices when parseable (not just
  mass-gap matching, which was vulnerable to aliases like G+V ≈ R).
- Charge-state variants of the masked group are labeled as
  ``charge_variant_leakage``.
- Novel-chemistry matches on unannotated peaks surface their extended
  type instead of the catch-all ``unannotated``.
"""

import numpy as np

from instanovo_fm.eval.embed_eval_tasks.ig_attribution_helper import (
    AMINO_ACID_MASSES,
    CHARGE_VARIANT_TOLERANCE_DA,
    PROTON_MASS,
    SHIFT_NULL_DISTANCES,
    TOP_K_RANKED_PEAKS_EMIT,
    WATER_MASS,
    FragmentGroup,
    STRUCTURAL_CATEGORIES,
    _gap_residues_from_sequence,
    _parse_series_info,
    compute_topk_attribution_analysis,
    detect_charge_variant_indices,
    expected_mz_at_other_charges,
)


# ---------------------------------------------------------------------------
# Small helpers for building synthetic fixtures
# ---------------------------------------------------------------------------

def _aa_mass(r):
    """Return the monoisotopic mass of an amino-acid letter.

    ``AMINO_ACID_MASSES`` keys L and I under the shared "L/I" entry (they're
    isobaric for b/y calculations); normalize here so tests can write plain
    single-letter sequences.
    """
    if r in ("L", "I"):
        return AMINO_ACID_MASSES["L/I"]
    return AMINO_ACID_MASSES[r]


def _b_ion_mz(residues, i, charge=1):
    """Singly/multiply protonated b_i m/z for the first ``i`` residues."""
    neutral = sum(_aa_mass(r) for r in residues[:i])
    return (neutral + charge * PROTON_MASS) / charge


def _y_ion_mz(residues, j, charge=1):
    """Singly/multiply protonated y_j m/z for the last ``j`` residues."""
    neutral = sum(_aa_mass(r) for r in residues[-j:]) + WATER_MASS
    return (neutral + charge * PROTON_MASS) / charge


def _neutral_mass(residues):
    return sum(_aa_mass(r) for r in residues) + WATER_MASS


def _make_groups(peak_specs):
    """Build FragmentGroup objects from ``[(group_key, peak_idx, ion_type, mz), ...]``."""
    groups = []
    for key, peak_idx, ion_type, mz in peak_specs:
        groups.append(FragmentGroup(
            group_key=key, peak_indices=[peak_idx], base_idx=peak_idx,
            base_mz=mz, ion_type=ion_type,
        ))
    return groups


# ---------------------------------------------------------------------------
# Series-info parser and gap-residue resolver
# ---------------------------------------------------------------------------

class TestParseSeriesInfo:
    def test_basic_b_ion(self):
        assert _parse_series_info("b3+") == ("b", 3, 1)

    def test_basic_y_ion(self):
        assert _parse_series_info("y11++") == ("y", 11, 2)

    def test_isotope_suffix_is_stripped(self):
        assert _parse_series_info("y10+[+1]") == ("y", 10, 1)

    def test_loss_suffix_is_stripped(self):
        assert _parse_series_info("b6+-H2O") == ("b", 6, 1)

    def test_precursor_returns_none(self):
        assert _parse_series_info("p^2") is None

    def test_custom_label_returns_none(self):
        assert _parse_series_info("custom:immonium_Tyr@136.0759") is None

    def test_empty_returns_none(self):
        assert _parse_series_info("") is None
        assert _parse_series_info(None) is None


class TestGapResiduesFromSequence:
    def test_b_ion_gap(self):
        # VPRPVTEK: b6 covers VPRPVT; b7 covers VPRPVTE; gap = E
        residues = list("VPRPVTEK")
        assert _gap_residues_from_sequence(residues, "b", 6, 7) == ["E"]

    def test_y_ion_gap(self):
        # HPFVDSNLLYQFR: y10 spans VDSNLLYQFR; y11 spans FVDSNLLYQFR; gap = F
        residues = list("HPFVDSNLLYQFR")
        assert _gap_residues_from_sequence(residues, "y", 10, 11) == ["F"]

    def test_multi_residue_y_gap(self):
        # y3 → y1 skips G,V (residues just N-terminal of y1)
        residues = list("RPEVDGVR")
        gap = _gap_residues_from_sequence(residues, "y", 3, 1)
        # y1 covers R; y3 covers GVR; skipped residues = GV
        assert gap == ["G", "V"]

    def test_order_independent(self):
        residues = list("VPRPVTEK")
        assert (_gap_residues_from_sequence(residues, "b", 6, 7)
                == _gap_residues_from_sequence(residues, "b", 7, 6))


# ---------------------------------------------------------------------------
# Charge-variant detection
# ---------------------------------------------------------------------------

class TestDetectChargeVariantIndices:
    def test_detects_doubly_charged_variant_of_singly_charged(self):
        residues = list("SIEAVHEDIR")
        # y5+ covers HEDIR
        y5_plus = _y_ion_mz(residues, 5, charge=1)
        y5_pp = _y_ion_mz(residues, 5, charge=2)
        # Build a spectrum with only those two peaks
        mz = np.array([y5_plus, y5_pp])
        variants = detect_charge_variant_indices(
            masked_base_mz=y5_plus, masked_charge=1, peak_mz_array=mz,
        )
        # Masked peak is at index 0, variant (y5++) at index 1
        assert 1 in variants
        assert 0 not in variants  # the masked peak isn't counted as its own variant

    def test_no_variants_when_peaks_are_unrelated(self):
        mz = np.array([100.0, 200.0, 300.0])
        assert detect_charge_variant_indices(
            masked_base_mz=500.0, masked_charge=1, peak_mz_array=mz,
        ) == set()

    def test_peak_exactly_at_tolerance_boundary_is_flagged(self):
        """Regression for hero_003: a z=2 +1 isotope sits exactly 0.5 Da
        off the expected z=2 m/z (ISOTOPE_SPACING ≈ 1.003, /2 ≈ 0.502).
        Strict ``<`` missed it; ``<=`` catches it.
        """
        # Masked y5+ (z=1) — expected z=2 at (neutral + 2p)/2
        residues = list("SIEAVHEDIR")
        y5_plus = _y_ion_mz(residues, 5, charge=1)
        y5_pp = _y_ion_mz(residues, 5, charge=2)
        # A synthetic peak exactly 0.5 Da off from y5_pp
        exactly_at_boundary = y5_pp + 0.5
        mz = np.array([y5_plus, y5_pp, exactly_at_boundary])
        variants = detect_charge_variant_indices(
            masked_base_mz=y5_plus, masked_charge=1, peak_mz_array=mz,
            tol_da=0.5,
        )
        # Both the z=2 base (1) and the boundary peak (2) must be flagged
        assert 1 in variants
        assert 2 in variants

    def test_sibling_masked_peak_variants_are_flagged(self):
        """Regression for hero_003: y8+-H2O is a sibling in the masked
        group, and its z=2 variant y8++-H2O was classified as unannotated
        because the detector only looked at the base y8+ m/z. Feeding
        sibling m/z via ``extra_masked_mz`` now catches it.
        """
        residues = list("GYFPIHLAAWK")
        y8_plus = _y_ion_mz(residues, 8, charge=1)
        # y8+-H2O = y8+ minus water
        y8_plus_loss = y8_plus - WATER_MASS
        # y8++-H2O = z=2 of y8+-H2O sibling
        sibling_neutral = y8_plus_loss - PROTON_MASS
        y8_pp_loss = (sibling_neutral + 2 * PROTON_MASS) / 2
        mz = np.array([y8_plus, y8_plus_loss, y8_pp_loss])

        # Without sibling input: base-only detection misses the loss variant
        without_siblings = detect_charge_variant_indices(
            masked_base_mz=y8_plus, masked_charge=1, peak_mz_array=mz,
        )
        assert 2 not in without_siblings

        # With sibling input: the z=2 loss variant is flagged
        with_siblings = detect_charge_variant_indices(
            masked_base_mz=y8_plus, masked_charge=1, peak_mz_array=mz,
            extra_masked_mz=[y8_plus_loss],
        )
        assert 2 in with_siblings


# ---------------------------------------------------------------------------
# Complementary-pair regression — the original bug test
# ---------------------------------------------------------------------------

class TestComplementaryPair:
    """The motivating bug: ``_classify_peak`` previously compared the
    neutral-mass sum against ``precursor_mass + WATER_MASS``, double-
    counting the terminal water. Every real complement was ~18 Da off
    the check and was demoted to ``opposite_series``.
    """

    def _build_vprpvtek_fixture(self):
        residues = list("VPRPVTEK")
        precursor_neutral = _neutral_mass(residues)
        b6_mz = _b_ion_mz(residues, 6, charge=1)
        y2_mz = _y_ion_mz(residues, 2, charge=1)
        # Include a y11-ish peak that is clearly NOT the complement — should
        # remain opposite_series.
        y1_mz = _y_ion_mz(residues, 1, charge=1)

        groups = _make_groups([
            ("b6+", 0, "b", b6_mz),
            ("y2+", 1, "y", y2_mz),
            ("y1+", 2, "y", y1_mz),
        ])
        annotations = ["b6+", "y2+", "y1+"]
        mz = np.array([b6_mz, y2_mz, y1_mz])
        attributions = np.array([0.0, 1.0, 0.5])  # y2+ is top-1
        masked_group = groups[0]
        return (attributions, masked_group, groups, annotations, mz,
                precursor_neutral)

    def test_y2_of_b6_is_complementary_pair(self):
        attributions, masked_group, groups, annotations, mz, prec_mass = \
            self._build_vprpvtek_fixture()
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked_group,
            all_groups=groups,
            annotations=annotations,
            mz=mz,
            precursor_mass=prec_mass,
        )
        # y2+ is the complement of b6+ for an 8-mer (i=6, L-i=2).
        assert topk.top1_annotation == "y2+"
        assert topk.top1_category == "complementary_pair"

    def test_non_complement_y_ion_is_opposite_series(self):
        attributions, masked_group, groups, annotations, mz, prec_mass = \
            self._build_vprpvtek_fixture()
        # Make y1+ the top-attributed peak instead of y2+
        attributions = np.array([0.0, 0.3, 1.0])
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked_group,
            all_groups=groups,
            annotations=annotations,
            mz=mz,
            precursor_mass=prec_mass,
        )
        assert topk.top1_annotation == "y1+"
        assert topk.top1_category == "opposite_series"


# ---------------------------------------------------------------------------
# Series-index ladder classifier (avoids mass-coincidence aliasing)
# ---------------------------------------------------------------------------

class TestLadderByPositionIndex:
    """Use the RPEVDGVR / y3+ → y1+ case from hero_008: the mass gap
    (G + V = 156.09 Da) happens to match R (156.10 Da) — the old mass-only
    path mis-labeled y1+ as a ladder_neighbor. Series-index logic
    correctly flags it as near_ladder.
    """

    def _build_rpevdgvr_fixture(self):
        residues = list("RPEVDGVR")
        prec_mass = _neutral_mass(residues)
        y3_mz = _y_ion_mz(residues, 3, charge=1)
        y1_mz = _y_ion_mz(residues, 1, charge=1)
        y2_mz = _y_ion_mz(residues, 2, charge=1)
        y4_mz = _y_ion_mz(residues, 4, charge=1)
        groups = _make_groups([
            ("y3+", 0, "y", y3_mz),
            ("y1+", 1, "y", y1_mz),
            ("y2+", 2, "y", y2_mz),
            ("y4+", 3, "y", y4_mz),
        ])
        annotations = ["y3+", "y1+", "y2+", "y4+"]
        mz = np.array([y3_mz, y1_mz, y2_mz, y4_mz])
        masked = groups[0]
        return annotations, mz, groups, masked, prec_mass

    def test_y1_of_y3_is_near_ladder_not_ladder_neighbor(self):
        ann, mz, groups, masked, prec_mass = self._build_rpevdgvr_fixture()
        # Only y1+ attributed — force it to top
        attributions = np.array([0.0, 1.0, 0.0, 0.0])
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked,
            all_groups=groups,
            annotations=ann,
            mz=mz,
            precursor_mass=prec_mass,
        )
        assert topk.top1_annotation == "y1+"
        # Gap is 2 positions → near_ladder. Without the series-index fix
        # this would be ladder_neighbor (mass-coincidence alias).
        assert topk.top1_category == "near_ladder"

    def test_y2_of_y3_is_ladder_neighbor(self):
        ann, mz, groups, masked, prec_mass = self._build_rpevdgvr_fixture()
        attributions = np.array([0.0, 0.0, 1.0, 0.0])  # y2+ on top
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=ann, mz=mz,
            precursor_mass=prec_mass,
        )
        assert topk.top1_annotation == "y2+"
        assert topk.top1_category == "ladder_neighbor"

    def test_y4_of_y3_is_ladder_neighbor(self):
        ann, mz, groups, masked, prec_mass = self._build_rpevdgvr_fixture()
        attributions = np.array([0.0, 0.0, 0.0, 1.0])  # y4+ on top
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=ann, mz=mz,
            precursor_mass=prec_mass,
        )
        assert topk.top1_annotation == "y4+"
        assert topk.top1_category == "ladder_neighbor"


# ---------------------------------------------------------------------------
# Charge-variant leakage takes priority over annotation
# ---------------------------------------------------------------------------

class TestChargeVariantCategory:
    def test_doubly_charged_variant_is_leakage(self):
        residues = list("SIEAVHEDIR")
        prec_mass = _neutral_mass(residues)
        y5_plus = _y_ion_mz(residues, 5, charge=1)
        y5_pp = _y_ion_mz(residues, 5, charge=2)
        y6_plus = _y_ion_mz(residues, 6, charge=1)

        groups = _make_groups([
            ("y5+", 0, "y", y5_plus),
            ("y5++", 1, "y", y5_pp),  # the charge variant (still annotated)
            ("y6+", 2, "y", y6_plus),
        ])
        annotations = ["y5+", "y5++", "y6+"]
        mz = np.array([y5_plus, y5_pp, y6_plus])
        attributions = np.array([0.0, 1.0, 0.2])  # y5++ attributed most
        masked = groups[0]
        # Caller is expected to feed detect_charge_variant_indices output in
        charge_variants = detect_charge_variant_indices(
            masked_base_mz=y5_plus, masked_charge=1, peak_mz_array=mz,
        )
        charge_variants.difference_update(set(masked.peak_indices))
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=annotations,
            mz=mz, precursor_mass=prec_mass,
            charge_variant_indices=charge_variants,
        )
        assert topk.top1_annotation == "y5++"
        assert topk.top1_category == "charge_variant_leakage"
        # ranked_peaks must also carry the leakage category (regression
        # against the previous bug where ranked_peaks overrode the category
        # to "unannotated" for non-group entries).
        ranked = {rp.peak_idx: rp for rp in topk.ranked_peaks}
        assert ranked[1].category == "charge_variant_leakage"
        assert ranked[1].is_group is False


# ---------------------------------------------------------------------------
# Novel-chemistry lookup re-labels unannotated peaks
# ---------------------------------------------------------------------------

class TestNovelChemLookup:
    def test_unannotated_internal_fragment_gets_typed(self):
        # Simplified fixture: mask a b-ion, include one unannotated peak
        # that the novel-chemistry probe has already matched as an internal
        # fragment. Without the lookup the category would be "unannotated".
        residues = list("ABCDEFGH")  # dummy peptide (not used for chemistry)
        b3_mz = 350.0
        unann_mz = 200.0
        groups = _make_groups([
            ("b3+", 0, "b", b3_mz),
        ])
        # unannotated peak has no group — it just exists in the mz array
        annotations = ["b3+", None]
        mz = np.array([b3_mz, unann_mz])
        attributions = np.array([0.0, 1.0])  # unannotated peak is top-1
        masked = groups[0]

        novel_lookup = {1: "internal_fragment"}
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=annotations,
            mz=mz, precursor_mass=1000.0,
            novel_chem_lookup=novel_lookup,
        )
        assert topk.top1_category == "internal_fragment"

    def test_ranked_peaks_carry_extended_chem_category(self):
        """Regression: the ``ranked_peaks`` list previously hardcoded
        ``category="unannotated"`` for any peak without a group, silently
        dropping the novel-chemistry re-labeling that ``_classify_peak`` did.
        Now every ranked unannotated peak must carry its classified category.
        """
        b3_mz = 350.0
        unann_mz = 200.0
        groups = _make_groups([("b3+", 0, "b", b3_mz)])
        annotations = ["b3+", None]
        mz = np.array([b3_mz, unann_mz])
        attributions = np.array([0.0, 1.0])
        masked = groups[0]

        novel_lookup = {1: "internal_fragment"}
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=annotations,
            mz=mz, precursor_mass=1000.0,
            novel_chem_lookup=novel_lookup,
        )
        # The ranked_peaks entry for this peak must also carry the extended-
        # chemistry type, not "unannotated".
        ranked = {rp.peak_idx: rp for rp in topk.ranked_peaks}
        assert 1 in ranked
        assert ranked[1].category == "internal_fragment"

    def test_unannotated_without_lookup_stays_unannotated(self):
        b3_mz = 350.0
        unann_mz = 200.0
        groups = _make_groups([("b3+", 0, "b", b3_mz)])
        annotations = ["b3+", None]
        mz = np.array([b3_mz, unann_mz])
        attributions = np.array([0.0, 1.0])
        masked = groups[0]
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=masked, all_groups=groups, annotations=annotations,
            mz=mz, precursor_mass=1000.0,
        )
        assert topk.top1_category == "unannotated"
        # And the ranked_peaks entry should also be "unannotated".
        ranked = {rp.peak_idx: rp for rp in topk.ranked_peaks}
        assert ranked[1].category == "unannotated"


# ---------------------------------------------------------------------------
# topk_fractions covers all structural categories (no KeyErrors downstream)
# ---------------------------------------------------------------------------

class TestExpectedMzAtOtherCharges:
    """Single source of truth for charge-state m/z math — used by both
    :func:`detect_charge_variant_indices` and the hero-report charge-state
    variant block.
    """

    def test_base_charge_is_excluded(self):
        result = expected_mz_at_other_charges(500.0, base_charge=1, max_charge=4)
        assert set(result) == {2, 3, 4}

    def test_round_trip_preserves_neutral_mass(self):
        residues = list("SIEAVHEDIR")
        y5_plus = _y_ion_mz(residues, 5, charge=1)
        expected = expected_mz_at_other_charges(y5_plus, base_charge=1)
        # z=2 should equal the directly-computed y5++ m/z
        y5_pp = _y_ion_mz(residues, 5, charge=2)
        assert abs(expected[2] - y5_pp) < 1e-6

    def test_zero_inputs_return_empty(self):
        assert expected_mz_at_other_charges(0.0, 1) == {}
        assert expected_mz_at_other_charges(500.0, 0) == {}


class TestConstantsWiring:
    """Lock-in tests for the promoted module-level constants."""

    def test_shift_null_distances_are_positive_and_ordered(self):
        assert len(SHIFT_NULL_DISTANCES) == 3
        assert all(d > 0 for d in SHIFT_NULL_DISTANCES)
        assert list(SHIFT_NULL_DISTANCES) == sorted(SHIFT_NULL_DISTANCES)

    def test_charge_variant_tolerance_around_half_isotope(self):
        from instanovo_fm.eval.embed_eval_tasks.ig_attribution_helper import (
            ISOTOPE_SPACING,
        )
        # Tolerance is 0.5 Da — sits within ~2 mDa of ISOTOPE_SPACING/2
        # (≈0.5017), so with the detector's ``<=`` inequality a z=2 +1
        # isotope sitting exactly 0.5 Da off the base still registers.
        half_iso = ISOTOPE_SPACING / 2
        assert abs(CHARGE_VARIANT_TOLERANCE_DA - half_iso) < 0.01

    def test_top_k_emit_cap_is_reasonable(self):
        assert 5 <= TOP_K_RANKED_PEAKS_EMIT <= 50


class TestGroupLevelCategoryUsesBase:
    """Regression for the group-aggregation bug: in a y-group containing
    the base y2 ion plus a neutral-loss sibling (y2-NH3), the group-level
    ``RankedPeak.category`` must reflect the *base* ion's classification
    (``complementary_pair``) even when the loss sibling is encountered
    first in attribution rank and by itself would be ``opposite_series``.
    """

    def test_complement_group_label_from_base_even_when_loss_ranks_higher(self):
        residues = list("VPRPVTEK")
        prec_mass = _neutral_mass(residues)
        b6_mz = _b_ion_mz(residues, 6, charge=1)
        y2_mz = _y_ion_mz(residues, 2, charge=1)
        # y2-NH3: same group as y2 (parent_annotation = "y2+"), but the
        # neutral-mass sum with b6 is off by 17 Da so it'd classify as
        # opposite_series on its own.
        y2_nh3_mz = y2_mz - 17.026549  # NH3
        # Build groups manually: y2+ group contains base + loss sibling.
        b6_group = FragmentGroup(
            group_key="b6+", peak_indices=[0], base_idx=0, base_mz=b6_mz, ion_type="b",
        )
        y2_group = FragmentGroup(
            group_key="y2+", peak_indices=[1, 2], base_idx=1,  # base = y2+ (index 1)
            base_mz=y2_mz, ion_type="y",
        )
        groups = [b6_group, y2_group]
        annotations = ["b6+", "y2+", "y2+-NH3"]
        mz = np.array([b6_mz, y2_mz, y2_nh3_mz])
        # Attribution: loss sibling (idx 2) gets a *higher* per-peak IG
        # value than the base (idx 1), so it's encountered first in the
        # rank list. The old code would take its classification.
        attributions = np.array([0.0, 0.4, 1.0])

        topk = compute_topk_attribution_analysis(
            attributions=attributions, masked_group=b6_group,
            all_groups=groups, annotations=annotations, mz=mz,
            precursor_mass=prec_mass,
        )
        # Find the y2+ group entry in ranked_peaks
        y2_rp = next((rp for rp in topk.ranked_peaks if rp.annotation == "y2+"), None)
        assert y2_rp is not None, "y2+ group missing from ranked_peaks"
        # Must carry the base ion's classification, not the loss child's
        assert y2_rp.category == "complementary_pair", (
            f"Group inherited wrong category: {y2_rp.category} "
            "(should be complementary_pair from base)"
        )


class TestSequenceContextComplementFormula:
    """Regression for the second-site +WATER_MASS bug. The complement of
    b_i in peptide neutral-mass space is ``M_neutral − b_i_neutral`` — and
    for singly-charged ions ``y_{L-i}+ = M_neutral + 2·proton − b_i+``.
    No extra water belongs in that formula.
    """

    def test_complement_mz_matches_actual_y_ion(self):
        # VPRPVTEK: real y1+ is at 147.113 (the terminus K + water + proton)
        residues = list("VPRPVTEK")
        prec_mass = _neutral_mass(residues)
        b7_mz = _b_ion_mz(residues, 7, charge=1)
        expected_y1_mz = _y_ion_mz(residues, 1, charge=1)
        # This is the formula that should live in _compute_sequence_context
        # (no spurious +WATER_MASS)
        comp_mz = prec_mass + 2 * PROTON_MASS - b7_mz
        assert abs(comp_mz - expected_y1_mz) < 1e-4, (
            f"Complement formula off by {comp_mz - expected_y1_mz:.4f} Da "
            f"(got {comp_mz}, expected {expected_y1_mz})"
        )


class TestTopkFractionsCoverage:
    def test_every_structural_category_is_keyed(self):
        residues = list("VPRPVTEK")
        prec_mass = _neutral_mass(residues)
        b6_mz = _b_ion_mz(residues, 6, charge=1)
        y2_mz = _y_ion_mz(residues, 2, charge=1)
        groups = _make_groups([
            ("b6+", 0, "b", b6_mz),
            ("y2+", 1, "y", y2_mz),
        ])
        annotations = ["b6+", "y2+"]
        mz = np.array([b6_mz, y2_mz])
        attributions = np.array([0.0, 1.0])
        topk = compute_topk_attribution_analysis(
            attributions=attributions,
            masked_group=groups[0], all_groups=groups, annotations=annotations,
            mz=mz, precursor_mass=prec_mass,
        )
        for k_bucket in topk.topk_fractions.values():
            for cat in STRUCTURAL_CATEGORIES:
                assert cat in k_bucket
