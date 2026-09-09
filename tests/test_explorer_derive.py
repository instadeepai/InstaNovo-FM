"""The explorer's derived display columns.

Eight of the explorer's 37 fields are computed from free-text metadata rather than read
from a column, and the rules are not recoverable from the field names. The chemistry rules
in particular were reverse-engineered from the published explorer, so the two tests that
matter here replay the published level counts through them and require every class to come
back to the exact published total. A rule that is nearly right shows up as a handful of
spectra in the wrong class, which is precisely what those tests catch.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "explorer"))

import derive  # noqa: E402

# The published explorer's own `search_modifications` level counts, over its 100,000
# spectra. Ground truth for the two chemistry rules: they have to partition exactly this.
PUBLISHED_SEARCH_MODIFICATIONS: dict[str, int] = {
    "default (N-term acetylation, Met oxidation)": 81124,
    "SILAC": 7367,
    "UbiSite; Ubiquitination": 2584,
    "N-term pyroQ; N-term pyroE; Cysteinylation": 2454,
    "TMT18": 1285,
    "TMT6": 1126,
    "Ubiquitination": 951,
    "TMT10": 695,
    "N-glycosylation": 595,
    "Phosphorylation": 448,
    "TMT10; phosphorylation": 231,
    "N-glyco": 230,
    "R-citrullin": 217,
    "phosphorylation": 129,
    "Y-phospho": 94,
    "iTRAQ4": 49,
    "K-methyl": 41,
    "K-succinyl": 38,
    "K-propionyl": 38,
    "K-acetyl": 35,
    "K-butyryl": 33,
    "K-trimethyl": 32,
    "R-dimethyl": 30,
    "K-dimethyl": 30,
    "K-crotonyl": 28,
    "K-hydroisobutyryl": 28,
    "K-formyl": 26,
    "K-glutaryl": 25,
    "K-malonyl": 16,
    "R-methyl": 12,
    "K-biotinyl": 8,
    "P-hydroxyproline": 1,
}

PUBLISHED_LABEL_CHEM = {
    "label-free": 88888,
    "SILAC": 7367,
    "TMT 6/10-plex": 2052,
    "TMTpro 18-plex": 1285,
    "isobaric (unspecified)": 359,
    "iTRAQ 4-plex": 49,
}

PUBLISHED_ENRICHMENT = {
    "none": 94101,
    "ubiquitination (GG)": 3535,
    "phosphorylation": 902,
    "N-glycosylation": 825,
    "acyl / methyl PTM": 420,
    "citrullination": 217,
}

# `search_quant == "TMT"` over the same spectra. iTRAQ is quantified on the precursor,
# so it is not part of this total -- which is the whole reason the residual works out.
PUBLISHED_TMT_QUANT_ROWS = 3696


def test_enrichment_partitions_the_published_counts_exactly() -> None:
    """Every enrichment class must come back to its published total."""
    got: Counter[str] = Counter()
    for label, n in PUBLISHED_SEARCH_MODIFICATIONS.items():
        got[derive.enrichment(label)] += n
    assert dict(got) == PUBLISHED_ENRICHMENT


def test_label_chem_partitions_the_published_counts_exactly() -> None:
    """Same for labelling chemistry, including the isobaric-without-a-named-reagent class.

    Those rows carry no reagent in their modification label, so they can only be found
    through `search_quant`; the residual between the TMT-quant row count and the
    TMT-named modification rows is exactly that class.
    """
    got: Counter[str] = Counter()
    for label, n in PUBLISHED_SEARCH_MODIFICATIONS.items():
        got[derive.label_chem(label, "precursor")] += n

    named_tmt = got["TMT 6/10-plex"] + got["TMTpro 18-plex"]
    unnamed = PUBLISHED_TMT_QUANT_ROWS - named_tmt
    assert unnamed == 359, "the TMT residual is the isobaric-unspecified class"
    got["label-free"] -= unnamed
    got["isobaric (unspecified)"] += unnamed

    assert dict(got) == PUBLISHED_LABEL_CHEM


def test_hydroxyproline_is_not_an_acyl_or_methyl_mark() -> None:
    """It is neither, and the published explorer counts it as "none"."""
    assert derive.enrichment("P-hydroxyproline") == "none"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        # Supplemental activation: both tokens present, so EThcD -- not ETD, not HCD.
        ("FTMS + c NSI d sa Full ms2 732.8905@etd120.83@hcd28.00 [100.0000-1476.0000]", "EThcD"),
        ("ITMS + c NSI t d sa Full ms2 680.3209@etd70.00@hcd25.00 [120.0000-1371.0000]", "EThcD"),
        # A lone etd token is ETD.
        ("FTMS + c NSI d Full ms2 415.5768@etd57.37 [100.0000-1257.0000]", "ETD"),
        ("ITMS + c NSI r d Full ms2 459.2382@hcd28.00 [100.0000-929.0000]", "HCD"),
        ("ITMS + c NSI d Full ms2 500.0000@cid35.00 [100.0000-1000.0000]", "CID"),
        ("FTMS + p NSI Full ms1 [350.0000-1400.0000]", "not stated"),
        (None, "not stated"),
    ],
)
def test_activation_reads_the_filter_line(header: str | None, expected: str) -> None:
    """Activation comes from the @method tokens, because frag_type conflates EThcD."""
    assert derive.activation(header) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("ITMS + c NSI r d Full ms2 459.2382@hcd28.00 [100.0000-929.0000]", "Ion trap (ITMS)"),
        ("FTMS + c NSI d Full ms2 738.7092@hcd27.00 [100.0000-2280.0000]", "Orbitrap (FTMS)"),
        ("TOF + something", "other / not stated"),
        (None, "other / not stated"),
    ],
)
def test_analyser_reads_the_filter_line(header: str | None, expected: str) -> None:
    """The analyser is the token the filter line opens with."""
    assert derive.analyser(header) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("FTMS + c NSI d Full ms2 738.70@hcd27.00 [100.0000-2280.0000]", 100.0),
        # Some runs write fewer decimals, and some start above 100.
        ("FTMS + p NSI d Full ms2 1221.60@hcd28.00 [100.00-2510.00]", 100.0),
        ("ITMS + c NSI t d sa Full ms2 680.32@etd70.00@hcd25.00 [120.0000-1371.0000]", 120.0),
        ("no scan range here", None),
    ],
)
def test_ms2_low_mz_reads_the_scan_range(header: str, expected: float | None) -> None:
    """The low end of the [low-high] range that closes the filter line."""
    got = derive.ms2_low_mz(header)
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == expected


@pytest.mark.parametrize(
    ("charge", "expected"),
    [
        ("2", "2+"),
        (3, "3+"),
        ("0", "not reported (0)"),
        (0, "not reported (0)"),
        (None, "not reported (0)"),
        ("", "not reported (0)"),
        ("7", "7+"),
    ],
)
def test_charge_cat(charge: object, expected: str) -> None:
    """Zero means the search reported no charge, which is not the same as 0+."""
    assert derive.charge_cat(charge) == expected


def test_sequence_length_ignores_unimod_annotations() -> None:
    """Peptide length is residues, so the bracketed modification tags do not count."""
    assert derive.sequence_length("PEPTIDE") == 7.0
    assert derive.sequence_length("LC[UNIMOD:4]YVALDFEQEM") == 12.0
    assert derive.sequence_length("[UNIMOD:1]PEPTIDEK") == 8.0
    assert np.isnan(derive.sequence_length(None))


def test_replicate_peptide_is_recomputed_from_the_rows_given() -> None:
    """ "One of the ten most repeated" is a claim about this point set, so rank it here."""
    seqs = np.array(["A"] * 5 + ["B"] * 4 + ["C"] * 3 + [f"u{i}" for i in range(20)], dtype=object)
    out = derive.replicate_peptide(seqs, top=2)
    assert set(out[:5]) == {"A"}
    assert set(out[5:9]) == {"B"}
    # C did not make the top 2, so it is unlabelled despite being a duplicate.
    assert set(out[9:12]) == {derive.NO_DUPLICATE}


def test_replicate_peptide_breaks_ties_deterministically() -> None:
    """Equal counts must not resolve by dict order, or two builds disagree."""
    seqs = np.array(["B", "B", "A", "A", "C"], dtype=object)
    first = derive.replicate_peptide(seqs, top=1)
    assert set(first) == {"A", derive.NO_DUPLICATE}, "alphabetical wins a count tie"


def test_to_float_maps_the_written_out_missing_values_to_nan() -> None:
    """Several numeric columns arrive as object dtype with "nan"/"None" spelled out."""
    got = derive.to_float(
        np.array(["1.5", "nan", "None", "", None, b"2.5", 3, "bogus"], dtype=object)
    )
    assert got[0] == 1.5
    assert got[5] == 2.5
    assert got[6] == 3.0
    assert np.isnan(got[[1, 2, 3, 4, 7]]).all()


def test_every_field_in_the_spec_has_a_label_and_group() -> None:
    """The UI groups the menus by these, so a missing one renders as an empty heading."""
    for name, spec in {**derive.NUM_FIELDS, **derive.CAT_FIELDS}.items():
        assert spec.get("label"), f"{name} has no label"
        assert spec.get("group"), f"{name} has no group"


def test_the_spec_matches_the_published_field_count() -> None:
    """14 numeric and 23 categorical, as the published explorer ships."""
    assert len(derive.NUM_FIELDS) == 14
    assert len(derive.CAT_FIELDS) == 23
