"""Derive the explorer's display columns from the eval harness's per-spectrum metadata.

The explorer shows 14 numeric and 23 categorical fields. Most come straight out of
``embeddings.h5``, but eight are derived here, and the rules are not guessable from the
field names alone:

``analyser``, ``activation``, ``ms2_low_mz``
    Parsed out of the Thermo filter line in ``header``, e.g.
    ``FTMS + c NSI d sa Full ms2 732.89@etd120.83@hcd28.00 [100.0000-1476.0000]``.
    The stored ``frag_type`` column is not usable for this: its ``ETD`` class mixes EThcD
    with ETD and its ``HCID`` class mixes EThcD with HCD, so the filter line is the only
    honest source.

``label_chem``, ``enrichment``
    Read out of the free-text ``search_modifications`` label, with ``search_quant`` as the
    fallback for isobaric runs whose modification label does not name a reagent. The rules
    below were checked against the published explorer's own level counts and reproduce
    every one of them exactly.

``charge_cat``, ``ptm``, ``replicate_peptide``
    Presentation recodings of ``precursor_charge``, ``ptm_present`` and a top-N duplicate
    ranking.

One deliberate difference from the published explorer. It splits the ETD-family spectra
688 EThcD / 181 ETD; the rule here -- supplemental activation, i.e. a filter line carrying
both ``@etd`` and ``@hcd`` -- splits the same family about 806/84. That is correct Thermo
semantics, so this does not reproduce the published split and is not trying to; the two
also describe different 100,000-row samples, which accounts for some of the gap.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Field spec. Labels and groups match the published explorer so the UI reads the same.
# ---------------------------------------------------------------------------

NUM_FIELDS: dict[str, dict[str, str]] = {
    "collision_energy": {"label": "Collision energy (NCE)", "group": "Acquisition physics"},
    "ms2_low_mz": {"label": "MS2 scan-range start (m/z)", "group": "Acquisition physics"},
    "spectrum_confidence": {"label": "Model spectrum confidence", "group": "Spectrum quality"},
    "hyperscore": {"label": "Hyperscore", "group": "Spectrum quality"},
    "precursor_mz": {"label": "Precursor m/z", "group": "Peptide & spectrum"},
    "precursor_mass": {"label": "Precursor mass (Da)", "group": "Peptide & spectrum"},
    "sequence_length": {"label": "Peptide length", "group": "Peptide & spectrum"},
    "hydrophobicity": {"label": "Hydrophobicity (GRAVY)", "group": "Peptide & spectrum"},
    "retention_time": {"label": "Retention time (s)", "group": "Peptide & spectrum"},
    "delta_mass": {"label": "Precursor mass error (Da)", "group": "Spectrum quality"},
    "expectation": {"label": "Search expectation value", "group": "Spectrum quality"},
    "nextscore": {"label": "Runner-up hyperscore", "group": "Spectrum quality"},
    "probability": {"label": "PSM probability", "group": "Spectrum quality"},
    "precursor_intensity": {"label": "Precursor intensity", "group": "Spectrum quality"},
}

CAT_FIELDS: dict[str, dict[str, str]] = {
    "analyser": {"label": "Mass analyser (from filter line)", "group": "Acquisition physics"},
    "activation": {"label": "Activation (from filter line)", "group": "Acquisition physics"},
    "search_fragmentation": {"label": "Search fragmentation", "group": "Acquisition physics"},
    "frag_type": {"label": "Fragmentation (as stored)", "group": "Acquisition physics"},
    "search_detector": {"label": "Detector", "group": "Acquisition physics"},
    "search_instrument": {"label": "Instrument", "group": "Acquisition physics"},
    "acquisition": {"label": "Acquisition mode", "group": "Acquisition physics"},
    "label_chem": {"label": "Labelling chemistry", "group": "Sample chemistry"},
    "enrichment": {"label": "Enrichment / PTM focus", "group": "Sample chemistry"},
    "search_quant": {"label": "Quantification", "group": "Sample chemistry"},
    "modification_class": {"label": "Modification class", "group": "Sample chemistry"},
    "modification_types": {"label": "Modification types", "group": "Sample chemistry"},
    "glyco_class": {"label": "Glycan class", "group": "Sample chemistry"},
    "ptm": {"label": "Modified?", "group": "Sample chemistry"},
    "search_modifications": {"label": "Search modifications", "group": "Sample chemistry"},
    "search_enzyme": {"label": "Enzyme", "group": "Sample chemistry"},
    "search_organism": {"label": "Organism", "group": "Biology & provenance"},
    "search_project": {"label": "PRIDE project", "group": "Biology & provenance"},
    "experiment_name": {"label": "Experiment / run", "group": "Biology & provenance"},
    "protein": {"label": "Protein", "group": "Biology & provenance"},
    "charge_cat": {"label": "Precursor charge", "group": "Peptide & spectrum"},
    "sequence": {"label": "Peptide sequence", "group": "Peptide & spectrum"},
    "replicate_peptide": {"label": "Top-10 repeated peptide", "group": "Peptide & spectrum"},
}

# Fields this module computes rather than copies.
DERIVED = ("analyser", "activation", "ms2_low_mz", "label_chem", "enrichment", "charge_cat",
           "ptm", "replicate_peptide", "sequence_length")

NOT_STATED = "other / not stated"
TOP_DUPLICATES = 10
NO_DUPLICATE = "(not a top-10 duplicate)"

# ---------------------------------------------------------------------------
# Thermo filter line
# ---------------------------------------------------------------------------

_ACTIVATION = re.compile(r"@([a-z]+)")
_SCAN_RANGE = re.compile(r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]")
_ANNOTATION = re.compile(r"\[[^\]]*\]")
_RESIDUE = re.compile(r"[A-Za-z]")


def analyser(header: str | None) -> str:
    """Mass analyser from the token that opens a Thermo filter line."""
    if not isinstance(header, str):
        return NOT_STATED
    if header.startswith("FTMS"):
        return "Orbitrap (FTMS)"
    if header.startswith("ITMS"):
        return "Ion trap (ITMS)"
    return NOT_STATED


def activation(header: str | None) -> str:
    """Activation type from the ``@method`` tokens in a Thermo filter line.

    Two tokens means supplemental activation: ``@etd...@hcd...`` is EThcD, which is a
    different experiment from either ETD or HCD alone and has to be its own class.
    """
    if not isinstance(header, str):
        return "not stated"
    methods = _ACTIVATION.findall(header)
    if not methods:
        return "not stated"
    if "etd" in methods:
        return "EThcD" if "hcd" in methods else "ETD"
    if "hcd" in methods:
        return "HCD"
    if "cid" in methods:
        return "CID"
    return "not stated"


def ms2_low_mz(header: str | None) -> float:
    """Low end of the MS2 scan range, ``[low-high]`` at the end of the filter line."""
    if not isinstance(header, str):
        return np.nan
    m = _SCAN_RANGE.search(header)
    return float(m.group(1)) if m else np.nan


# ---------------------------------------------------------------------------
# Sample chemistry, out of the free-text search_modifications label
# ---------------------------------------------------------------------------

# Ordered: the first match wins, so the more specific reagent is tested before the
# generic family. TMT18 must precede TMT6/TMT10 because "TMT18" contains neither but a
# substring test for "TMT" would swallow it.
_LABEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SILAC", ("silac",)),
    ("TMTpro 18-plex", ("tmt18", "tmtpro")),
    ("TMT 6/10-plex", ("tmt6", "tmt10", "tmt2", "tmt")),
    ("iTRAQ 4-plex", ("itraq4", "itraq")),
)

_ENRICHMENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ubiquitination (GG)", ("ubisite", "ubiquitin")),
    ("phosphorylation", ("phospho",)),
    ("N-glycosylation", ("n-glyco", "n-glycosylation")),
    ("citrullination", ("citrullin",)),
    # Every K-/R- acyl and methyl mark, kept as one class because individually they are
    # a few dozen spectra each. Hydroxyproline is deliberately not here: it is neither an
    # acyl nor a methyl mark, and the published explorer counts it as "none".
    ("acyl / methyl PTM", (
        "k-methyl", "k-dimethyl", "k-trimethyl", "r-methyl", "r-dimethyl",
        "k-acetyl", "k-succinyl", "k-propionyl", "k-butyryl", "k-crotonyl",
        "k-hydroisobutyryl", "k-formyl", "k-glutaryl", "k-malonyl", "k-biotinyl",
    )),
)


def label_chem(search_modifications: str | None, search_quant: str | None) -> str:
    """Labelling chemistry, from the modification label with quant as the fallback.

    An isobaric run whose modification label names no reagent is still isobaric --
    ``search_quant`` is what catches those, and calling them label-free would be wrong.
    """
    mods = (search_modifications or "").lower()
    for name, needles in _LABEL_RULES:
        if any(x in mods for x in needles):
            return name
    if (search_quant or "").strip().lower() in {"tmt", "itraq", "isobaric"}:
        return "isobaric (unspecified)"
    return "label-free"


def enrichment(search_modifications: str | None) -> str:
    """What the sample was enriched for, if anything."""
    mods = (search_modifications or "").lower()
    for name, needles in _ENRICHMENT_RULES:
        if any(x in mods for x in needles):
            return name
    return "none"


# ---------------------------------------------------------------------------
# Presentation recodings
# ---------------------------------------------------------------------------


def charge_cat(charge: Any) -> str:
    """Precursor charge as a label. Zero means the search did not report one."""
    try:
        z = int(float(charge))
    except (TypeError, ValueError):
        return "not reported (0)"
    return "not reported (0)" if z <= 0 else f"{z}+"


def ptm(ptm_present: Any) -> str:
    """Whether the peptide carries any modification."""
    try:
        return "modified" if int(ptm_present) else "unmodified"
    except (TypeError, ValueError):
        return "unmodified"


def replicate_peptide(sequences: np.ndarray, top: int = TOP_DUPLICATES) -> np.ndarray:
    """Label each row with its peptide, for the ``top`` most repeated peptides only.

    Recomputed from the rows being displayed rather than carried over: "one of the ten
    most repeated peptides" is a statement about this point set, so importing a ranking
    from a different one would label the wrong spectra.
    """
    counts = Counter(sequences.tolist())
    # Rank by frequency, then by sequence, so ties do not depend on dict order.
    ranked = [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]
    winners = set(ranked)
    return np.array([s if s in winners else NO_DUPLICATE for s in sequences], dtype=object)


def sequence_length(sequence: str | None) -> float:
    """Residue count: UNIMOD annotations removed, then the remaining residue letters counted.

    Counting whatever is left after stripping the brackets is not enough. A terminal
    modification is written ``[UNIMOD:1]-PEPTIDER``, and that separating hyphen would
    score as a residue -- which is how this first disagreed with the stored
    ``sequence_length`` on 2,961 of 100,000 rows, every one of them terminally modified.
    """
    if not isinstance(sequence, str):
        return np.nan
    return float(len(_RESIDUE.findall(_ANNOTATION.sub("", sequence))))


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------

_MISSING = {"", "nan", "none", "null", "na", "n/a", "-"}


def to_float(values: np.ndarray) -> np.ndarray:
    """Coerce a metadata column to float64, mapping the string sentinels to NaN.

    Several numeric columns arrive as HDF5 ``object`` -- i.e. strings, sometimes with
    ``"nan"``/``"None"`` written out -- so ``astype(float)`` alone raises on them.
    """
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        if v is None or isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, str):
            if v.strip().lower() in _MISSING:
                continue
            try:
                out[i] = float(v)
            except ValueError:
                continue
        else:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                continue
    return out
