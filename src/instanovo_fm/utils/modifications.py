"""Utility functions for extracting modification types from peptide sequences.

This module provides functions to parse modified peptide sequences and extract
the types of post-translational modifications (PTMs) present. It supports:
- UniMod format ([UNIMOD:X])
- Pyteomics parsing
- Regex-based pattern matching for common modifications
- Mass shift detection and mapping to known modifications
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np


def clean_peptide_sequence(peptide: str) -> str:
    """Clean peptide sequence by removing modifications and keeping only standard amino acids.
    
    Args:
        peptide: Modified peptide sequence
        
    Returns:
        Cleaned sequence with only standard amino acids
    """
    if not peptide:
        return ""
    
    # Remove common modifications
    peptide = re.sub(r'\[UNIMOD:\d+\]', '', peptide)
    
    # Remove other common modifications
    modifications = [
        r'\(ox\)', r'\(ph\)', r'\(ac\)', r'\(me\)', r'\(gly\)', r'\(glc\)',
        r'\(\+[0-9.]+\)', r'\(\-[0-9.]+\)'
    ]
    
    for mod in modifications:
        peptide = re.sub(mod, '', peptide, flags=re.IGNORECASE)
    
    # Keep only standard amino acid letters
    clean_pep = ''.join(c for c in peptide if c.isalpha() and c.upper() in 'ACDEFGHIKLMNPQRSTVWY')
    
    return clean_pep


def extract_modification_types_regex(sequence: str) -> List[str]:
    """Extract modification types from sequence using regex patterns as a fallback.
    
    This function uses pattern matching to identify modifications in peptide sequences.
    It supports:
    - UniMod IDs ([UNIMOD:X])
    - Common modification abbreviations (ox, ph, ac, etc.)
    - Textual modification names (Oxidation, Phosphorylation, etc.)
    - Mass shift patterns ((+15.99), (-18.01), etc.)
    
    Args:
        sequence: Modified peptide sequence
        
    Returns:
        List of modification types found (deduplicated)
    """
    mod_types: List[str] = []

    # Helper: map common mass deltas (Da) to human-readable names
    def map_mass_to_name(mass_delta: float) -> str | None:
        known_masses = {
            15.9949: 'Oxidation',
            79.9663: 'Phospho',
            57.0215: 'Carbamidomethyl',
            42.0106: 'Acetyl',
            14.0157: 'Methyl',
            28.0313: 'Dimethyl',
            42.04695: 'Trimethyl',
            0.9840: 'Deamidated',
            -17.0265: 'Ammonia-loss',
            -18.0106: 'Dehydrated',
            203.0794: 'HexNAc',
            114.0429: 'GlyGly',
        }
        for ref_mass, name in known_masses.items():
            if abs(mass_delta - ref_mass) <= 0.01:
                return name
        return None

    # Common modification patterns
    modification_patterns = {
        r'\[UNIMOD:(\d+)\]': 'UniMod',  # UniMod format [UNIMOD:X]
        r'\(UniMod:(\d+)\)': 'UniMod',  # PyOpenMS format (UniMod:X)
        r'\[(\d+)\]': 'Mass_shift',  # Generic mass shift in brackets [43], [136], etc.
        r'\(ox\)': 'Oxidation',
        r'\(ph\)': 'Phosphorylation',
        r'\(ac\)': 'Acetylation',
        r'\(me\)': 'Methylation',
        r'\(gly\)': 'Glycosylation',
        r'\(glc\)': 'Glucosylation',
    }

    # First, detect textual modification names that may appear without brackets
    textual_patterns = [
        (r'(?i)Oxid(?:ation)?', 'Oxidation'),
        (r'(?i)Phospho|Phosphorylation', 'Phospho'),
        (r'(?i)Carbamidomethyl', 'Carbamidomethyl'),
        (r'(?i)Acetyl', 'Acetyl'),
        (r'(?i)Methyl', 'Methyl'),
        (r'(?i)Deamidated', 'Deamidated'),
        (r'(?i)HexNAc', 'HexNAc'),
        (r'(?i)GlyGly', 'GlyGly'),
        (r'(?i)Dehydrated', 'Dehydrated'),
        (r'(?i)Ammonia[- ]loss', 'Ammonia-loss'),
        (r'(?i)Glu->pyro-Glu|Glu→pyro-Glu|pyro-Glu', 'Glu->pyro-Glu'),
    ]
    for pat, name in textual_patterns:
        if re.search(pat, sequence):
            mod_types.append(name)

    # UniMod and shorthand patterns
    for pattern, mod_name in modification_patterns.items():
        matches = re.findall(pattern, sequence, re.IGNORECASE)
        if matches:
            if mod_name == 'UniMod':
                # Map common UniMod IDs to names
                for match in matches:
                    uni_id = int(match)
                    uni_mappings = {
                        1: 'Acetyl',
                        4: 'Carbamidomethyl',
                        5: 'Carbamyl',
                        6: 'Carbamidomethyl',  # various Cys protection IDs
                        7: 'Deamidated',
                        21: 'Phospho',
                        35: 'Oxidation',
                        36: 'Dehydrated',
                        37: 'Oxidation',
                        38: 'Oxidation',
                        39: 'Oxidation',
                        40: 'Oxidation',
                        259: 'Glu->pyro-Glu',
                        267: 'Glu->pyro-Glu',
                        385: 'Ammonia-loss',
                        481: 'Glu->pyro-Glu',
                    }
                    mod_types.append(uni_mappings.get(uni_id, f'UniMod:{uni_id}'))
            else:
                mod_types.extend([mod_name] * len(matches))

    # Generic mass shift patterns like (+15.99) or (-18.01)
    mass_matches = re.findall(r'\(([+-])\s*([0-9]*\.?[0-9]+)\)', sequence)
    for sign, mass_str in mass_matches:
        try:
            mass_val = float(mass_str)
            if sign == '-':
                mass_val = -mass_val
            mapped = map_mass_to_name(mass_val)
            if mapped:
                mod_types.append(mapped)
            else:
                # Keep a generic descriptor to not lose information
                mod_types.append('Mass_shift' if mass_val >= 0 else 'Mass_loss')
        except Exception:
            continue

    return list(set(mod_types))


def extract_modification_types_pyteomics(sequence: str) -> List[str]:
    """Extract modification types from sequence using pyteomics.
    
    This function uses the pyteomics library for more robust parsing of
    modified peptide sequences. Falls back to regex parsing if pyteomics
    is unavailable or parsing fails.
    
    Args:
        sequence: Modified peptide sequence
        
    Returns:
        List of modification types found
    """
    try:
        from pyteomics import parser
        
        # Try different parsing approaches for pyteomics
        try:
            # First try standard parsing
            parsed = parser.parse(sequence)
        except Exception:
            # If standard parsing fails, try to convert UniMod format to pyteomics format
            # Convert [UNIMOD:x] to (UniMod:x) format
            converted_seq = re.sub(r'\[UNIMOD:(\d+)\]', r'(UniMod:\1)', sequence)
            try:
                parsed = parser.parse(converted_seq)
            except Exception:
                # If both approaches fail, fall back to regex
                return extract_modification_types_regex(sequence)
        
        # Extract modification types from the parsed sequence
        mod_types = []
        for item in parsed:
            # Handle both tuple/list (aa, mod) and simple string tokens
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                aa, mod = item[0], item[1]
                if mod and isinstance(mod, dict) and 'unimod' in mod:
                    try:
                        # Try to get UniMod name for the modification
                        import importlib
                        unimod_module = importlib.import_module('pyteomics.unimod')
                        mod_name = unimod_module.by_id[mod['unimod']]['title']
                        mod_types.append(mod_name)
                    except (ImportError, KeyError, TypeError):
                        # If UniMod ID not found or pyteomics.unimod not available, use a generic name
                        mod_types.append(f"UniMod:{mod['unimod']}")
            # else: ignore tokens that don't carry modification info
        
        return mod_types
        
    except ImportError:
        # Fallback to regex method
        return extract_modification_types_regex(sequence)
    except Exception:
        # Fallback to regex method for any other errors
        return extract_modification_types_regex(sequence)


def extract_modification_types(sequence: str) -> List[str]:
    """Extract modification types from a peptide sequence.
    
    This is the main entry point for modification extraction. It tries
    pyteomics first for robust parsing, then falls back to regex patterns.
    
    Args:
        sequence: Modified peptide sequence
        
    Returns:
        List of unique modification types found (sorted alphabetically)
    """
    if not sequence or not isinstance(sequence, str) or len(sequence.strip()) == 0:
        return []
    
    # Try pyteomics-assisted extraction first
    mod_types = extract_modification_types_pyteomics(sequence)
    
    # If none found via pyteomics, fall back to regex heuristics
    if not mod_types:
        mod_types = extract_modification_types_regex(sequence)
    
    # Return unique, sorted modification types
    return sorted(set(mod_types)) if mod_types else []


def compute_modification_types(
    peptides: np.ndarray,
    use_modified_peptide: bool = True
) -> np.ndarray:
    """Compute modification types for an array of peptide sequences.
    
    This function processes multiple peptide sequences and extracts their
    modification types. It returns a human-readable string for each peptide
    combining all modifications found.
    
    Args:
        peptides: Array of peptide sequences (can be modified or unmodified)
        use_modified_peptide: Whether the input contains modification annotations
        
    Returns:
        Array of modification type strings. Each entry is either:
        - 'Unmodified' if no modifications found
        - A string like 'Oxidation' or 'Oxidation + Phospho' for modified peptides
        - 'Unknown' if the peptide is invalid/empty
    """
    if peptides is None or len(peptides) == 0:
        return np.array([])
    
    modification_types = []
    
    for peptide in peptides:
        # Handle invalid peptides (None, empty string, or the string 'None')
        if (peptide is None or 
            not isinstance(peptide, str) or 
            len(peptide.strip()) == 0 or 
            peptide.strip().lower() == 'none'):
            modification_types.append('Unmodified')
            continue
        
        # Only extract modifications if we're using modified peptide format
        if use_modified_peptide:
            try:
                mod_types = extract_modification_types(peptide)
                
                if mod_types:
                    # Join multiple modifications with ' + '
                    modification_types.append(' + '.join(mod_types))
                else:
                    modification_types.append('Unmodified')
            except Exception:
                # If extraction fails, mark as unmodified
                modification_types.append('Unmodified')
        else:
            # If not using modified peptide format, all are unmodified
            modification_types.append('Unmodified')
    
    return np.array(modification_types, dtype=object)


# ---------------------------------------------------------------------------
# Per-modification binary detectors (phospho / glyco)
#
# Used to build binary linear-probe targets (mod_phospho / mod_glyco). Handles
# both encodings seen in the FM corpora:
#   - UNIMOD tokens, e.g. "[UNIMOD:21]" (LCFM `sequence` column), and
#   - rounded total residue-mass tokens, e.g. "S[167]" = Ser+phospho, or a
#     composite N-glycan carried as one big delta on Asn e.g. "N[1330]"
#     (HCFM `modified_peptide_unimod` column).
#
# PHOSPHO = UNIMOD:21.
# GLYCO_UNIMOD_IDS = every UNIMOD accession classified as *glycosylation*
#   (N-linked / O-linked / Other), extracted from unimod.xml (2026-07-02;
#   450 records, of which 71 appear in LCFM). Shipped as a static frozenset so
#   the eval has no runtime UNIMOD dependency.
# ---------------------------------------------------------------------------

PHOSPHO_UNIMOD_IDS = frozenset({21})

GLYCO_UNIMOD_IDS = frozenset({
    41, 43, 54, 137, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 213, 295, 305, 307, 308, 309, 310,
    311, 393, 408, 428, 429, 454, 490, 512, 793, 907, 910, 1286, 1303, 1304,
    1367, 1375, 1376, 1377, 1378, 1379, 1400, 1408, 1409, 1410, 1411, 1412,
    1413, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435,
    1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447,
    1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459,
    1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471,
    1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483,
    1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495,
    1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507,
    1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519,
    1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531,
    1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543,
    1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555,
    1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567,
    1568, 1570, 1571, 1572, 1573, 1575, 1577, 1578, 1579, 1580, 1581, 1582,
    1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594,
    1595, 1596, 1597, 1598, 1599, 1600, 1602, 1604, 1606, 1607, 1608, 1609,
    1610, 1611, 1612, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622,
    1623, 1624, 1625, 1626, 1627, 1628, 1630, 1631, 1632, 1633, 1634, 1635,
    1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647,
    1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659,
    1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671,
    1672, 1673, 1674, 1675, 1676, 1678, 1679, 1680, 1681, 1682, 1683, 1684,
    1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696,
    1697, 1698, 1699, 1700, 1701, 1702, 1703, 1705, 1706, 1707, 1708, 1709,
    1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722,
    1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1732, 1733, 1735, 1736,
    1737, 1738, 1739, 1740, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749,
    1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761,
    1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773,
    1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785,
    1786, 1840, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939,
    1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951,
    1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
    1964, 1965, 1966, 1967, 1968, 1969, 2022, 2028, 2029,
})

# Nominal total-mass tokens for phospho on S/T/Y (residue-mass encoding).
_PHOSPHO_RESMASS_TOKENS = ("S[167]", "T[181]", "Y[243]")
# Residue base masses for composite-glycan detection in the residue-mass format.
_GLYCO_RES_BASE = {"N": 114.043, "S": 87.032, "T": 101.048}
# Only used for the residue-mass *fallback* encoding (e.g. HCFM). Threshold sits
# above all common peptide-labeling reagents (TMTpro ~304, iTRAQ8 ~304) so those
# are not mistaken for glyco; genuine N-glycan compositions are far heavier
# (>=~800 Da). LCFM is UNIMOD-encoded, where glyco is detected exactly by ID and
# this heuristic is irrelevant. Small O-glycans in the residue-mass format are
# not caught by this fallback (they are, correctly, in the UNIMOD path).
_GLYCO_COMPOSITE_MIN_DELTA = 500.0

_UNIMOD_ID_RE = re.compile(r"(?i)unimod:(\d+)")
_RESMASS_NST_RE = re.compile(r"([NST])\[(\d+(?:\.\d+)?)\]")


def _unimod_ids(sequence: str) -> set:
    """Return the set of integer UNIMOD IDs referenced in a sequence string."""
    return {int(m) for m in _UNIMOD_ID_RE.findall(sequence)}


def has_phosphorylation(sequence: str) -> bool:
    """True if the (modified) peptide carries a phosphorylation.

    Detects both UNIMOD:21 and the rounded residue-mass tokens S[167]/T[181]/
    Y[243], plus common textual/shorthand forms.
    """
    if not sequence or not isinstance(sequence, str):
        return False
    if _unimod_ids(sequence) & PHOSPHO_UNIMOD_IDS:
        return True
    if any(tok in sequence for tok in _PHOSPHO_RESMASS_TOKENS):
        return True
    return bool(re.search(r"(?i)phospho|\(ph\)", sequence))


def has_glycosylation(sequence: str) -> bool:
    """True if the (modified) peptide carries a glycosylation.

    Detects any UNIMOD ID classified as glycosylation, composite glycans in the
    residue-mass encoding (a large delta on N/S/T, e.g. N[1330]), and common
    textual/shorthand forms (HexNAc, glyco, (gly)/(glc)).
    """
    if not sequence or not isinstance(sequence, str):
        return False
    if _unimod_ids(sequence) & GLYCO_UNIMOD_IDS:
        return True
    for res, mass in _RESMASS_NST_RE.findall(sequence):
        if float(mass) - _GLYCO_RES_BASE[res] > _GLYCO_COMPOSITE_MIN_DELTA:
            return True
    return bool(re.search(r"(?i)hexnac|glyco|\(gly\)|\(glc\)", sequence))


def compute_ptm_binary_flags(peptides: np.ndarray) -> dict:
    """Vectorized per-modification binary flags for an array of peptides.

    Returns a dict with int32 arrays ``mod_phospho`` and ``mod_glyco`` (1 if the
    peptide carries that modification, else 0), aligned to ``peptides``. Intended
    as binary linear-probe targets alongside ``ptm_present``.
    """
    if peptides is None or len(peptides) == 0:
        return {"mod_phospho": np.array([], dtype=np.int32),
                "mod_glyco": np.array([], dtype=np.int32)}
    phospho = np.fromiter(
        (has_phosphorylation(p) for p in peptides), dtype=np.int32, count=len(peptides)
    )
    glyco = np.fromiter(
        (has_glycosylation(p) for p in peptides), dtype=np.int32, count=len(peptides)
    )
    return {"mod_phospho": phospho, "mod_glyco": glyco}


# ---------------------------------------------------------------------------
# Deamidation of asparagine (N) — the isobaric-with-aspartate (D) case
#
# Deamidation (UNIMOD:7, +0.984 Da) on N converts it to a residue that is
# mass-identical to aspartate (D), a well-known source of ambiguity for de novo
# sequencing. Unlike glycosylation (which adds a large, easily-detected precursor
# mass shift), N-deamidation is a stringent test of whether the embedding
# captures fine spectral evidence rather than gross precursor mass.
#
# Detected residue-aware: the modification token must sit on an N. UNIMOD form
# "N[UNIMOD:7]"/"N(UniMod:7)"; rounded residue-mass form "N[115]"
# (Asn 114.043 + 0.984 = 115.027 -> 115; unambiguous, as aspartate is the letter D).
# ---------------------------------------------------------------------------

DEAMIDATION_UNIMOD_IDS = frozenset({7})
_DEAM_N_RE = re.compile(r"(?i)N[\[(]unimod:7[\])]|N\[115(?:\.\d+)?\]")


def has_deamidation_n(sequence: str) -> bool:
    """True if the peptide carries a deamidation on an asparagine (N)."""
    if not sequence or not isinstance(sequence, str):
        return False
    return bool(_DEAM_N_RE.search(sequence))


# ---------------------------------------------------------------------------
# Glycan subtype (family) classification for the glyco-differentiation probe
#
# For glyco-positive peptides, assign the glycan to a family so a multiclass
# probe can test whether the embedding differentiates glycans (not merely
# detects their presence). Mapping built from the UNIMOD compositions of the
# glycosylation-classified accessions present in LCFM (unimod.xml, 2026-07):
# priority sialylated (NeuAc/NeuGc) > fucosylated (dHex) > high-mannose
# (Hex5-9HexNAc2) > paucimannose (Hex<=3 HexNAc2) > complex/hybrid.
# ---------------------------------------------------------------------------

GLYCO_ID_FAMILY = {
    43: "complex_hybrid", 137: "high_mannose", 148: "paucimannose", 159: "paucimannose",
    305: "fucosylated", 307: "fucosylated", 308: "fucosylated", 309: "complex_hybrid",
    310: "complex_hybrid", 311: "complex_hybrid", 1408: "sialylated", 1409: "sialylated",
    1410: "sialylated", 1452: "high_mannose", 1462: "fucosylated", 1465: "high_mannose",
    1467: "fucosylated", 1468: "complex_hybrid", 1477: "fucosylated", 1480: "high_mannose",
    1481: "fucosylated", 1484: "fucosylated", 1487: "complex_hybrid", 1488: "sialylated",
    1496: "complex_hybrid", 1500: "fucosylated", 1504: "high_mannose", 1506: "sialylated",
    1509: "fucosylated", 1510: "sialylated", 1511: "fucosylated", 1519: "fucosylated",
    1529: "sialylated", 1531: "high_mannose", 1532: "complex_hybrid", 1534: "sialylated",
    1537: "fucosylated", 1540: "complex_hybrid", 1541: "fucosylated", 1547: "fucosylated",
    1549: "complex_hybrid", 1551: "sialylated", 1555: "fucosylated", 1559: "complex_hybrid",
    1562: "fucosylated", 1722: "fucosylated", 1746: "fucosylated", 1761: "fucosylated",
    1763: "complex_hybrid", 1766: "fucosylated", 1768: "fucosylated", 1769: "complex_hybrid",
    1771: "fucosylated", 1772: "complex_hybrid", 1773: "sialylated", 1775: "fucosylated",
    1776: "complex_hybrid", 1777: "sialylated", 1778: "fucosylated", 1779: "complex_hybrid",
    1780: "complex_hybrid", 1781: "fucosylated", 1782: "sialylated", 1783: "fucosylated",
    1784: "sialylated", 1785: "fucosylated", 1840: "fucosylated", 1963: "fucosylated",
    1964: "sialylated", 2028: "sialylated", 2029: "complex_hybrid",
}


def glyco_class(sequence: str) -> str:
    """Return the glycan family for a glyco-positive peptide, else ``'none'``.

    If several glyco accessions are present, the highest-priority family among
    them is returned (sialylated > fucosylated > complex_hybrid > high_mannose
    > paucimannose).
    """
    if not sequence or not isinstance(sequence, str):
        return "none"
    ids = _unimod_ids(sequence) & GLYCO_UNIMOD_IDS
    fams = {GLYCO_ID_FAMILY.get(i, "complex_hybrid") for i in ids}
    if not fams:
        # fall back to the presence detector (residue-mass / textual glyco)
        return "glyco_other" if has_glycosylation(sequence) else "none"
    for fam in ("sialylated", "fucosylated", "complex_hybrid", "high_mannose", "paucimannose"):
        if fam in fams:
            return fam
    return "complex_hybrid"


def compute_glyco_deam_targets(peptides: np.ndarray) -> dict:
    """Vectorized extra probe targets: ``mod_deam_n`` (binary) and ``glyco_class``.

    ``mod_deam_n``: 1 if the peptide carries a deamidation on N, else 0.
    ``glyco_class``: glycan family string for glyco-positive peptides, else
    ``'none'`` (excluded when the glyco-differentiation probe filters to
    glyco-positive spectra).
    """
    if peptides is None or len(peptides) == 0:
        return {"mod_deam_n": np.array([], dtype=np.int32),
                "glyco_class": np.array([], dtype=object)}
    deam_n = np.fromiter(
        (has_deamidation_n(p) for p in peptides), dtype=np.int32, count=len(peptides)
    )
    gclass = np.array([glyco_class(p) for p in peptides], dtype=object)
    return {"mod_deam_n": deam_n, "glyco_class": gclass}






