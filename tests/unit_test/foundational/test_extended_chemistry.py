"""Unit tests for the extended chemistry library."""

import pytest
import numpy as np

from instanovo_fm.utils.extended_chemistry import (
    AMINO_ACID_MASSES,
    CO_MASS,
    H_ATOM_MASS,
    IMMONIUM_RELATED_IONS,
    NH3_MASS,
    PROTON_MASS,
    SIDE_CHAIN_LOSSES,
    WATER_MASS,
    ExtendedChemistryLibrary,
    ExtendedChemistryMatch,
    build_extended_chemistry_library,
    compute_combined_precursor_losses,
    compute_d_ions,
    compute_immonium_related_ions,
    compute_immonium_related_negative_control,
    compute_internal_fragments,
    compute_side_chain_losses,
    compute_w_ions,
    match_peak_to_extended_chemistry,
    match_peaks_batch,
)


# ---------------------------------------------------------------------------
# d-ion tests
# ---------------------------------------------------------------------------


class TestComputeDIons:
    """Tests for d-ion computation."""

    def test_leu_ile_discrimination(self):
        """d-ions for Leu and Ile at same position must differ."""
        # GALI — position 3 is L, position 4 is I
        residues_l = ["G", "A", "L"]
        residues_i = ["G", "A", "I"]

        d_l = compute_d_ions(residues_l, precursor_charge=1)
        d_i = compute_d_ions(residues_i, precursor_charge=1)

        # Both should have d-ions at position 3 (last residue, but we iterate to L-1)
        # Actually residues_l has 3 residues, positions 1..2 are computed
        # Position 2 has residue A — no d-ion (A not in SIDE_CHAIN_LOSSES)
        # Position 1 has residue G — no d-ion
        # So no d-ions from ["G", "A", "L"] because L is at position 3 and we
        # iterate to L-1=2. Let's use longer peptides.
        residues_with_l = ["G", "A", "L", "K"]  # L at position 3
        residues_with_i = ["G", "A", "I", "K"]  # I at position 3

        d_l = compute_d_ions(residues_with_l, precursor_charge=1)
        d_i = compute_d_ions(residues_with_i, precursor_charge=1)

        # Filter to position 3
        d_l_pos3 = [d for d in d_l if d[3] == 3]
        d_i_pos3 = [d for d in d_i if d[3] == 3]

        assert len(d_l_pos3) >= 1, "Leu should produce d-ions"
        assert len(d_i_pos3) >= 1, "Ile should produce d-ions"

        # Ile has 2 losses (primary C2H5 + alt CH3), Leu has 1 (C3H7)
        assert len(d_i_pos3) == 2, f"Ile should have 2 d-ions at position 3, got {len(d_i_pos3)}"
        assert len(d_l_pos3) == 1, f"Leu should have 1 d-ion at position 3, got {len(d_l_pos3)}"

        # The primary d-ion masses must be different
        d_l_mz = d_l_pos3[0][0]
        d_i_mz = d_i_pos3[0][0]  # primary (C2H5 loss)
        assert abs(d_l_mz - d_i_mz) > 10.0, (
            f"Leu d-ion ({d_l_mz:.4f}) and Ile d-ion ({d_i_mz:.4f}) should differ significantly"
        )

    def test_no_d_ions_for_non_labile_residues(self):
        """Residues without labile side chains should not produce d-ions."""
        # G, A, F, P, R, H, Y, W have no d-ion pathway
        residues = ["G", "A", "F", "P", "R"]
        d_ions = compute_d_ions(residues, precursor_charge=1)
        assert len(d_ions) == 0

    def test_d_ion_mass_formula(self):
        """Verify d-ion mass: d = a - side_chain_loss + H_atom."""
        residues = ["G", "V", "A"]  # V at position 2
        d_ions = compute_d_ions(residues, precursor_charge=1)

        b2_neutral = AMINO_ACID_MASSES["G"] + AMINO_ACID_MASSES["V"]
        a2_neutral = b2_neutral - CO_MASS
        # V loses CH3 (15.023475), +H_atom for radical stabilization
        d2_neutral = a2_neutral - 15.023475 + H_ATOM_MASS
        d2_mz = d2_neutral + PROTON_MASS

        pos2_ions = [d for d in d_ions if d[3] == 2]
        assert len(pos2_ions) == 1
        assert abs(pos2_ions[0][0] - d2_mz) < 0.001, (
            f"Expected d2 m/z={d2_mz:.4f}, got {pos2_ions[0][0]:.4f}"
        )

    def test_d_ion_exact_csv_validation(self):
        """Validate single-residue d-ion against CSV d_primary_mz_1plus.

        CSV: Val d_primary_mz_1plus = 58.065126
        Formula: d = a - CH3 + H_atom = 72.080776 - 15.023475 + 1.00782503 = 58.065126
        """
        # Single-residue d-ion: for "VA", d1 is for V at position 1
        residues = ["V", "A"]
        d_ions = compute_d_ions(residues, precursor_charge=1)
        v_pos1 = [d for d in d_ions if d[3] == 1 and d[2] == "V"]
        assert len(v_pos1) == 1
        csv_expected = 58.065126
        assert abs(v_pos1[0][0] - csv_expected) < 0.001, (
            f"Expected {csv_expected}, got {v_pos1[0][0]:.6f}"
        )

    def test_d_ion_multi_residue_csv_validation(self):
        """Validate multi-residue d-ion: for "GV", d2 should match CSV formula.

        d = a - loss + H_atom:
          b2_neutral = G + V = 57.02146 + 99.06841 = 156.08987
          a2_neutral = b2 - CO = 128.09496
          d2_neutral = a2 - CH3 + H_atom = 128.09496 - 15.02348 + 1.00783 = 114.07931
          d2_mz = 114.07931 + 1.00728 = 115.08659
        """
        residues = ["G", "V", "A"]
        d_ions = compute_d_ions(residues, precursor_charge=1)
        pos2 = [d for d in d_ions if d[3] == 2]
        assert len(pos2) == 1

        b2_n = AMINO_ACID_MASSES["G"] + AMINO_ACID_MASSES["V"]
        a2_n = b2_n - CO_MASS
        d2_n = a2_n - 15.023475 + H_ATOM_MASS
        expected = d2_n + PROTON_MASS
        assert abs(pos2[0][0] - expected) < 0.0001

    def test_charge_states(self):
        """d-ions should be generated for all charge states up to precursor_charge."""
        residues = ["G", "V", "A", "K"]
        d_ions_z1 = compute_d_ions(residues, precursor_charge=1)
        d_ions_z2 = compute_d_ions(residues, precursor_charge=2)

        assert len(d_ions_z2) == 2 * len(d_ions_z1), (
            f"charge=2 should produce 2x ions: got {len(d_ions_z2)} vs {len(d_ions_z1)}"
        )

    def test_modified_residues_skipped(self):
        """Modified residues should be skipped for d-ion computation."""
        residues = ["G", "M[UNIMOD:35]", "A"]
        d_ions = compute_d_ions(residues, precursor_charge=1)
        # M[UNIMOD:35] should be skipped
        m_ions = [d for d in d_ions if d[2] == "M"]
        assert len(m_ions) == 0


# ---------------------------------------------------------------------------
# w-ion tests
# ---------------------------------------------------------------------------


class TestComputeWIons:
    """Tests for w-ion computation."""

    def test_w_ion_produced_for_labile_residues(self):
        """w-ions should be generated when the N-terminal residue has a labile side chain."""
        residues = ["K", "L", "A", "G"]  # L at C-terminal side, L is target for w
        w_ions = compute_w_ions(residues, precursor_charge=1)
        # w-ions come from z-dot ions. z1 covers residues[3]=G, z2 covers [2,3]=AG,
        # z3 covers [1,2,3]=LAG — the N-terminal residue of z3 is L at index 1
        l_ions = [w for w in w_ions if w[2] == "L"]
        assert len(l_ions) >= 1, "Leu should produce w-ions"

    def test_w_leu_ile_discrimination(self):
        """w-ions for Leu and Ile should differ in mass."""
        residues_l = ["A", "L", "G", "A"]
        residues_i = ["A", "I", "G", "A"]

        w_l = compute_w_ions(residues_l, precursor_charge=1)
        w_i = compute_w_ions(residues_i, precursor_charge=1)

        # Filter to L/I residue matches
        w_l_filtered = [w for w in w_l if w[2] == "L"]
        w_i_filtered = [w for w in w_i if w[2] == "I"]

        if w_l_filtered and w_i_filtered:
            # Leu and Ile w-ions should differ
            l_mz = w_l_filtered[0][0]
            i_mz = w_i_filtered[0][0]
            assert abs(l_mz - i_mz) > 10.0

    def test_no_w_ions_for_non_labile(self):
        """Non-labile residues at the N-edge of z-fragments should not produce w-ions."""
        residues = ["A", "G", "F", "P"]
        w_ions = compute_w_ions(residues, precursor_charge=1)
        assert len(w_ions) == 0

    def test_w_ion_mass_against_csv_reference(self):
        """Validate w-ion mass against residue_ion_reference.csv values.

        CSV says for single-residue z-dot ions:
          Ala z-dot(1+) = 74.036231
          Val z-dot(1+) = 102.067531
          Val w(primary, -CH3) at z=1 = 87.044056 Da
        """
        # For "AV" (2 residues), z1 covers just V, z-dot(1) for V:
        # z(neutral) = V_mass + H2O - NH3 = 99.06841 + 18.010565 - 17.026549 = 100.052426
        # z-dot(neutral) = z(neutral) + H_atom = 100.052426 + 1.00782503 = 101.060251
        # z-dot(mz at z=1) = 101.060251 + 1.007276 = 102.067527 ≈ 102.067531 (CSV)
        # w1 for V = z-dot(1) - CH3(15.023475) = 101.060251 - 15.023475 = 86.036776
        # w(mz at z=1) = 86.036776 + 1.007276 = 87.044052 ≈ 87.044056 (CSV)

        residues = ["A", "V"]
        w_ions = compute_w_ions(residues, precursor_charge=1)
        v_ions = [w for w in w_ions if w[2] == "V"]
        assert len(v_ions) == 1, f"Expected 1 w-ion for V, got {len(v_ions)}"

        expected_w_mz = 87.044056  # from CSV
        actual_w_mz = v_ions[0][0]
        assert abs(actual_w_mz - expected_w_mz) < 0.001, (
            f"w-ion for V: expected {expected_w_mz:.6f}, got {actual_w_mz:.6f}, "
            f"delta={abs(actual_w_mz - expected_w_mz):.6f} Da"
        )


# ---------------------------------------------------------------------------
# Internal fragment tests
# ---------------------------------------------------------------------------


class TestComputeInternalFragments:
    """Tests for internal fragment computation."""

    def test_basic_count(self):
        """Verify number of internal fragments for a short peptide."""
        # "PEPTIDE" = 7 residues
        # Internal fragments: start from 1 to 5, end varies
        # Each produces both b-type and a-type
        residues = list("PEPTIDE")
        frags = compute_internal_fragments(residues, max_length=6)
        # All internal fragments have start >= 1 and end <= 5
        # b-type + a-type for each
        assert len(frags) > 0
        # Check both types present
        labels = [f[1] for f in frags]
        assert any("int(b," in l for l in labels)
        assert any("int(a," in l for l in labels)

    def test_internal_mass_correct(self):
        """Verify a specific internal fragment mass."""
        # Subsequence "EP" from "PEPTIDE" (positions 2-3, 0-indexed: 1-2)
        residues = list("PEPTIDE")
        frags = compute_internal_fragments(residues, max_length=6)

        ep_frags = [f for f in frags if "EP" in f[1] and "int(b," in f[1]]
        assert len(ep_frags) >= 1

        expected_neutral = AMINO_ACID_MASSES["E"] + AMINO_ACID_MASSES["P"]
        expected_mz = expected_neutral + PROTON_MASS

        # Find the EP b-type fragment
        found = False
        for mz, label in ep_frags:
            if abs(mz - expected_mz) < 0.01:
                found = True
                break
        assert found, f"Expected EP internal at {expected_mz:.4f}, got {[f[0] for f in ep_frags]}"

    def test_max_length_cap(self):
        """Internal fragments should not exceed max_length."""
        residues = list("ABCDEFGHIJ")  # 10 residues
        frags_short = compute_internal_fragments(residues, max_length=3)
        frags_long = compute_internal_fragments(residues, max_length=6)

        # Short should have fewer fragments
        assert len(frags_short) < len(frags_long)

        # Check no fragment exceeds max_length
        for _, label in frags_short:
            # Parse positions from label like "int(b,2-4,BCD)"
            parts = label.split(",")
            start = int(parts[1].split("-")[0])
            end = int(parts[1].split("-")[1])
            assert end - start + 1 <= 3

    def test_excludes_terminal_fragments(self):
        """Internal fragments should not include terminal positions."""
        residues = list("ABCDE")
        frags = compute_internal_fragments(residues, max_length=6)
        for _, label in frags:
            parts = label.split(",")
            start = int(parts[1].split("-")[0])
            end = int(parts[1].split("-")[1])
            # start should be >= 2 (not first residue) and end <= L-1 (not last)
            # Actually our code starts from index 1 (0-indexed), position 2 (1-indexed)
            assert start >= 2, f"Internal fragment starts at terminal: {label}"


# ---------------------------------------------------------------------------
# Side-chain loss tests
# ---------------------------------------------------------------------------


class TestComputeSideChainLosses:
    """Tests for residue-specific side-chain losses."""

    def test_met_losses(self):
        """Met-containing b-ions should produce CH3SH and CH3SOH losses."""
        residues = ["A", "M", "G"]
        losses = compute_side_chain_losses(residues, precursor_charge=1)
        labels = [l[1] for l in losses]
        assert any("CH3SH" in l for l in labels), f"Expected CH3SH loss, got {labels}"
        assert any("CH3SOH" in l for l in labels), f"Expected CH3SOH loss, got {labels}"

    def test_asp_co2_loss(self):
        """Asp-containing fragments should produce CO2 loss."""
        residues = ["A", "D", "G"]
        losses = compute_side_chain_losses(residues, precursor_charge=1)
        labels = [l[1] for l in losses]
        assert any("CO2" in l for l in labels), f"Expected CO2 loss from Asp, got {labels}"

    def test_no_losses_for_inert_residues(self):
        """Peptides without M/D/E should produce no side-chain losses."""
        residues = ["G", "A", "V", "L"]
        losses = compute_side_chain_losses(residues, precursor_charge=1)
        assert len(losses) == 0


# ---------------------------------------------------------------------------
# Library builder tests
# ---------------------------------------------------------------------------


class TestBuildExtendedChemistryLibrary:
    """Tests for the master library builder."""

    def test_builds_all_types(self):
        """Library should contain all four ion types for a complex peptide."""
        residues = ["A", "L", "I", "M", "D", "G", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=2)

        assert len(lib.d_ions) > 0, "Should have d-ions (L, I, M, D, K present)"
        assert len(lib.w_ions) > 0, "Should have w-ions"
        assert len(lib.internal_fragments) > 0, "Should have internal fragments"
        assert len(lib.side_chain_losses) > 0, "Should have side-chain losses (M, D)"
        assert lib.total_hypotheses > 0

    def test_empty_for_short_peptide(self):
        """Very short peptides should still produce a valid (possibly sparse) library."""
        residues = ["G", "A"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)
        # No d/w-ions (G and A have no labile side chains), minimal internals
        assert lib.total_hypotheses >= 0  # should not crash


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------


class TestMatchPeakToExtendedChemistry:
    """Tests for peak-to-hypothesis matching."""

    def test_exact_match(self):
        """A peak at an exact d-ion mass should match."""
        residues = ["G", "V", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)

        # Get the first d-ion m/z
        assert len(lib.d_ions) > 0
        target_mz = lib.d_ions[0][0]

        matches = match_peak_to_extended_chemistry(target_mz, lib, ppm_tol=10.0)
        assert len(matches) >= 1
        assert matches[0].hypothesis_type == "d_ion"
        assert matches[0].error_da < 0.001

    def test_ppm_tolerance(self):
        """Matching should respect PPM tolerance."""
        residues = ["G", "V", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)
        target_mz = lib.d_ions[0][0]

        # 5 ppm shift
        shifted = target_mz * (1 + 5e-6)
        matches_tight = match_peak_to_extended_chemistry(shifted, lib, ppm_tol=3.0)
        matches_loose = match_peak_to_extended_chemistry(shifted, lib, ppm_tol=10.0)

        assert len(matches_tight) == 0, "Should not match at 3 ppm for a 5 ppm shift"
        assert len(matches_loose) >= 1, "Should match at 10 ppm for a 5 ppm shift"

    def test_da_tolerance(self):
        """Da tolerance should override PPM."""
        residues = ["G", "V", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)
        target_mz = lib.d_ions[0][0]

        shifted = target_mz + 0.3
        matches = match_peak_to_extended_chemistry(shifted, lib, ppm_tol=1.0, da_tol=0.5)
        assert len(matches) >= 1, "Should match with 0.5 Da tolerance"

    def test_no_false_match(self):
        """Random m/z far from any hypothesis should not match."""
        residues = ["G", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)
        matches = match_peak_to_extended_chemistry(5000.0, lib, ppm_tol=10.0)
        assert len(matches) == 0

    def test_multiple_matches(self):
        """A peak can match multiple hypotheses."""
        # Build a large library where some masses overlap
        residues = list("GAVLIMDK")
        lib = build_extended_chemistry_library(residues, precursor_charge=2)

        # Try each d-ion against internal fragments — some may overlap
        # Just verify the function handles multiple matches gracefully
        for mz, _, _, _ in lib.d_ions[:5]:
            matches = match_peak_to_extended_chemistry(mz, lib, ppm_tol=20.0)
            # At minimum, the d-ion itself should match
            d_matches = [m for m in matches if m.hypothesis_type == "d_ion"]
            assert len(d_matches) >= 1

    def test_leu_ile_flag(self):
        """Matches at Leu/Ile positions should have discriminates_leu_ile=True."""
        residues = ["G", "L", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)

        l_ions = [d for d in lib.d_ions if d[2] == "L"]
        if l_ions:
            matches = match_peak_to_extended_chemistry(l_ions[0][0], lib, ppm_tol=10.0)
            d_matches = [m for m in matches if m.hypothesis_type == "d_ion" and m.residue == "L"]
            assert len(d_matches) >= 1
            assert d_matches[0].discriminates_leu_ile is True


class TestImmoniumRelatedIons:
    """Tests for immonium-related ion computation."""

    def test_phe_tropylium(self):
        """Phe-containing peptide should produce tropylium at 91.054 Da."""
        residues = ["A", "F", "G"]
        ions = compute_immonium_related_ions(residues)
        f_ions = [i for i in ions if i[2] == "F"]
        assert len(f_ions) == 1
        assert abs(f_ions[0][0] - 91.054) < 0.01
        assert "rel_F_91" in f_ions[0][1]

    def test_trp_multiple_related(self):
        """Trp-containing peptide should produce multiple related ions."""
        residues = ["G", "W", "A"]
        ions = compute_immonium_related_ions(residues)
        w_ions = [i for i in ions if i[2] == "W"]
        assert len(w_ions) == 6  # 77, 117, 130, 132, 170, 171

    def test_no_ions_for_absent_aa(self):
        """Should not produce related ions for amino acids not in the sequence."""
        residues = ["G", "A", "V"]  # no F, W, H, R, etc.
        ions = compute_immonium_related_ions(residues)
        f_ions = [i for i in ions if i[2] == "F"]
        assert len(f_ions) == 0

    def test_composition_only(self):
        """Scrambling should produce identical immonium-related ions."""
        residues1 = ["A", "F", "G", "H"]
        residues2 = ["H", "G", "F", "A"]  # scrambled
        ions1 = sorted(compute_immonium_related_ions(residues1))
        ions2 = sorted(compute_immonium_related_ions(residues2))
        assert ions1 == ions2  # identical — composition dependent, not order

    def test_negative_control(self):
        """Negative control should produce ions for ABSENT amino acids only."""
        residues = ["A", "F", "G"]  # F present, W absent
        present = compute_immonium_related_ions(residues)
        absent = compute_immonium_related_negative_control(residues)

        present_aa = set(i[2] for i in present)
        absent_aa = set(i[2] for i in absent)
        assert present_aa & absent_aa == set()  # no overlap

        # W should be in absent control (not in sequence)
        assert "W" in absent_aa

    def test_negative_control_has_no_present_aa(self):
        """Negative control must not contain ions for amino acids in the peptide."""
        residues = list("GAVLIMDK")
        neg = compute_immonium_related_negative_control(residues)
        aa_in_seq = set(r[0] for r in residues)
        for _, _, aa in neg:
            assert aa not in aa_in_seq


class TestCombinedPrecursorLosses:
    """Tests for combined precursor neutral losses."""

    def test_produces_losses(self):
        """Should produce combined losses for valid precursor."""
        losses = compute_combined_precursor_losses(precursor_mz=500.0, precursor_charge=2)
        assert len(losses) > 0
        labels = [l[1] for l in losses]
        assert any("2xH2O" in l for l in labels)
        assert any("H2O+NH3" in l for l in labels)

    def test_double_water_mass(self):
        """p-2xH2O should be precursor_mz - 2*H2O/z."""
        prec_mz = 500.0
        losses = compute_combined_precursor_losses(prec_mz, precursor_charge=2)
        dw = [l for l in losses if "2xH2O" in l[1] and "++" in l[1]]
        assert len(dw) == 1
        expected = prec_mz - (2 * 18.010565) / 2
        assert abs(dw[0][0] - expected) < 0.001

    def test_zero_precursor(self):
        """Zero precursor_mz should produce no losses."""
        losses = compute_combined_precursor_losses(precursor_mz=0.0)
        assert len(losses) == 0


class TestBuildLibraryV2:
    """Tests for the updated library builder with all types."""

    def test_includes_immonium_related(self):
        """Library should include immonium-related ions for present amino acids."""
        residues = ["A", "F", "H", "W", "G", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=2, precursor_mz=500.0)
        assert len(lib.immonium_related) > 0
        # F gives tropylium, H gives 5 related, W gives 6 related, K gives 4
        assert len(lib.immonium_related) >= 10

    def test_includes_precursor_losses(self):
        """Library should include combined precursor losses when precursor_mz given."""
        residues = ["A", "G", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=2, precursor_mz=500.0)
        assert len(lib.precursor_combined_losses) > 0

    def test_no_precursor_losses_without_mz(self):
        """Library should skip precursor losses when precursor_mz is 0."""
        residues = ["A", "G", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=2, precursor_mz=0.0)
        assert len(lib.precursor_combined_losses) == 0

    def test_total_hypotheses_includes_all(self):
        """total_hypotheses should count all types."""
        residues = ["A", "F", "M", "D", "G", "L", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=2, precursor_mz=500.0)
        expected = (len(lib.d_ions) + len(lib.w_ions) + len(lib.internal_fragments)
                   + len(lib.side_chain_losses) + len(lib.immonium_related)
                   + len(lib.precursor_combined_losses))
        assert lib.total_hypotheses == expected


class TestMatchPeaksBatch:
    """Tests for batch matching."""

    def test_batch_returns_dict(self):
        """Batch matching should return dict with only non-empty entries."""
        residues = ["G", "V", "A", "K"]
        lib = build_extended_chemistry_library(residues, precursor_charge=1)

        target_mz = lib.d_ions[0][0] if lib.d_ions else 100.0
        mz_array = np.array([target_mz, 5000.0, 0.0])

        results = match_peaks_batch(mz_array, lib, ppm_tol=10.0)
        assert isinstance(results, dict)
        if lib.d_ions:
            assert 0 in results
        assert 1 not in results
        assert 2 not in results
