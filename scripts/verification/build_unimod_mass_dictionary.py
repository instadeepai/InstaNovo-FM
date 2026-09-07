"""Build UNIMOD Mass Dictionary from Excel Files.

This script reads modification annotations from Excel files, queries the Pyteomics
UNIMOD database to discover amino acid specificities and masses, and generates:
1. A YAML mass dictionary with all observed modifications
2. A markdown validation report for proteomics practitioners
3. An Excel file of custom IN:xxx modifications requiring expert annotation
"""

import re
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Set, Tuple

import polars as pl
import typer
from lxml import etree
from pyteomics import mass

# Type aliases for clarity
ModificationInfo = Dict[str, Any]
SpecificityInfo = Dict[str, Any]

# Default paths
DEFAULT_GOLD_STANDARD = Path("mod_dicts/gold_standard_modifications.xlsx")
DEFAULT_AMBIGUOUS_MODS = Path("mod_dicts/PXD009449_ambiguous_mods.xlsx")
DEFAULT_OUTPUT_DIR = Path("mod_dicts")

# UNIMOD namespace for XML parsing
UNIMOD_NS = {"umod": "http://www.unimod.org/xmlns/schema/unimod_tables_1"}

# Position key mapping
POSITION_MAP = {
    "1": "Anywhere",
    "2": "Any N-term",
    "3": "Any N-term",
    "4": "Any C-term",
    "5": "Protein N-term",
    "6": "Protein C-term",
}

app = typer.Typer(help="Build UNIMOD mass dictionary from Excel modification files.")


def read_excel_files(
    gold_standard_path: Path, ambiguous_mods_path: Path
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Read both Excel files with modification data."""
    print("Reading Excel files...")

    gold_standard = pl.read_excel(gold_standard_path)
    ambiguous_mods = pl.read_excel(ambiguous_mods_path)

    # Normalize column names
    if "proposed encoding" in gold_standard.columns:
        gold_standard = gold_standard.rename(
            {"proposed encoding": "proposed_unimod_encoding"}
        )

    print(f"  - Gold standard: {len(gold_standard)} rows")
    print(f"  - Ambiguous mods: {len(ambiguous_mods)} rows")

    return gold_standard, ambiguous_mods


def separate_modification_types(
    df1: pl.DataFrame, df2: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Separate UNIMOD modifications from custom IN:xxx modifications."""
    print("\nSeparating modification types...")

    combined = pl.concat([df1, df2], how="diagonal")

    unimod_mods = combined.filter(
        pl.col("proposed_unimod_encoding").str.contains("UNIMOD:", literal=True)
    )
    custom_mods = combined.filter(
        pl.col("proposed_unimod_encoding").str.contains("IN:", literal=True)
    )

    print(f"  - UNIMOD modifications: {len(unimod_mods)} rows")
    print(f"  - Custom IN:xxx modifications: {len(custom_mods)} rows")

    return unimod_mods, custom_mods


def parse_encyclopedia_modification(mod_str: str) -> Tuple[bool, str, str]:
    """Parse encyclopeDIA modification notation, returning the amino acid, the encyclopedia modification ID, and whether the modification is terminal.

    Returns:
        (is_terminal, amino_acid, encyclopedia_modification_id)

    Examples:
        "K[156]" -> (False, "K", "156")
        "n[43]A" -> (True, "A", "43")
        "S[167]" -> (False, "S", "167")
    """
    # N-terminal modification: n[mass]AA
    n_term_match = re.match(r"n\[(\d+)\]([A-Z])", mod_str)
    if n_term_match:
        return True, n_term_match.group(2), n_term_match.group(1)

    # Regular modification: AA[mass]
    regular_match = re.match(r"([A-Z])\[(\d+)\]", mod_str)
    if regular_match:
        return False, regular_match.group(1), regular_match.group(2)

    return False, "", ""


def parse_unimod_encoding(encoding: str) -> int:
    """Parse UNIMOD encoding to extract the UNIMOD ID.

    The amino acid is read from the modification column, not from the encoding.

    Returns:
        unimod_id (0 if not found)

    Examples:
        "[UNIMOD:1]" -> 1
        "[UNIMOD:385]" -> 385
    """
    match = re.search(r"\[UNIMOD:(\d+)\]", encoding)
    if match:
        return int(match.group(1))
    return 0


def _process_single_row(row: Dict[str, Any], observed: Dict[int, Set[str]]) -> None:
    """Process a single row and update observed combinations."""
    modification = row.get("modification", "")
    encoding = row.get("proposed_unimod_encoding", "")

    if not encoding or not isinstance(encoding, str):
        return
    if not modification or not isinstance(modification, str):
        return

    is_terminal, aa_from_mod, _ = parse_encyclopedia_modification(modification)
    unimod_id = parse_unimod_encoding(encoding)

    if unimod_id <= 0:
        return

    if is_terminal:
        observed[unimod_id].add("")
    else:
        observed[unimod_id].add(aa_from_mod)


def _format_amino_acid_list(amino_acids: Set[str]) -> List[str]:
    """Format amino acids set for display, with N-term first if present."""
    if "" in amino_acids:
        return ["N-term"] + [a for a in sorted(amino_acids) if a]
    return sorted(amino_acids)


def extract_observed_combinations(unimod_df: pl.DataFrame) -> Dict[int, Set[str]]:
    """Extract which amino acids were actually observed with each UNIMOD ID.

    Returns:
        Dict mapping UNIMOD ID -> Set of amino acids (empty string for terminal)
    """
    print("\nExtracting observed amino acid + UNIMOD combinations...")

    observed: Dict[int, Set[str]] = defaultdict(set)

    for row in unimod_df.iter_rows(named=True):
        _process_single_row(row, observed)

    print(f"  - Found {len(observed)} unique UNIMOD IDs")
    for unimod_id, amino_acids in sorted(observed.items()):
        aa_list = _format_amino_acid_list(amino_acids)
        print(f"    - UNIMOD:{unimod_id}: {', '.join(aa_list)}")

    return dict(observed)


def _download_with_progress(url: str, destination: Path) -> None:
    """Download a file with a progress bar."""

    def report_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        sys.stdout.write(
            f"\r  Downloading: [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
        )
        sys.stdout.flush()

        if downloaded >= total_size:
            sys.stdout.write("\n")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, destination, reporthook=report_hook)


def _get_unimod_cache() -> Optional[Path]:
    """Set up and return the UNIMOD cache file path, downloading if needed."""
    cache_dir = Path(tempfile.gettempdir()) / "pyteomics_unimod"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "unimod.xml"

    if cache_file.exists():
        print("  - Using cached UNIMOD database")
        return cache_file

    print("  - Downloading UNIMOD database (first run only)...")
    unimod_url = "http://www.unimod.org/xml/unimod_tables.xml"
    try:
        _download_with_progress(unimod_url, cache_file)
        print(f"  - Cached to {cache_file}")
        return cache_file
    except Exception as e:
        print(f"  - ERROR: Could not download UNIMOD database: {e}")
        return None


def _parse_unimod_xml(cache_file: Path) -> Optional[etree._Element]:
    """Parse the UNIMOD XML file and return the root element."""
    try:
        print("  - Parsing XML database...")
        parser = etree.XMLParser(
            load_dtd=False,
            no_network=True,
            resolve_entities=False,
            remove_comments=True,
        )
        tree = etree.parse(str(cache_file), parser)
        print("  - Extracting modification data...")
        return tree.getroot()
    except Exception as e:
        print(f"  - ERROR: Could not parse UNIMOD XML: {e}")
        return None


def _find_modification_element(
    root: etree._Element, unimod_id: int
) -> Optional[etree._Element]:
    """Find modification element by record_id, trying with and without namespace."""
    mod_elem = root.find(
        f".//umod:modifications_row[@record_id='{unimod_id}']", UNIMOD_NS
    )
    if mod_elem is None:
        mod_elem = root.find(f".//modifications_row[@record_id='{unimod_id}']")
    return mod_elem


def _extract_required_attr(elem: etree._Element, attr: str, unimod_id: int) -> str:
    """Extract a required attribute from an element, raising if missing."""
    value = elem.get(attr)
    if not value:
        raise ValueError(f"Missing '{attr}' attribute for UNIMOD:{unimod_id}")
    return str(value)


def _parse_specificity_element(
    spec_elem: etree._Element, unimod_id: int
) -> SpecificityInfo:
    """Parse a single specificity element into a dict."""
    site = spec_elem.get("one_letter")
    if site is None:
        raise ValueError(
            f"Missing 'one_letter' attribute in specificity for UNIMOD:{unimod_id}"
        )

    position_key = _extract_required_attr(spec_elem, "position_key", unimod_id)

    hidden = spec_elem.get("hidden")
    if hidden is None:
        raise ValueError(
            f"Missing 'hidden' attribute in specificity for UNIMOD:{unimod_id}"
        )

    position = POSITION_MAP.get(position_key)
    if not position:
        raise ValueError(
            f"Unknown position_key '{position_key}' for UNIMOD:{unimod_id}"
        )

    return {
        "site": site,
        "position": position,
        "classification": "Unknown",
        "rare": hidden == "1",
    }


def _find_specificity_elements(
    root: etree._Element, unimod_id: int
) -> List[etree._Element]:
    """Find all specificity elements for a given modification."""
    spec_elems = root.findall(
        f".//umod:specificity_row[@mod_key='{unimod_id}']", UNIMOD_NS
    )
    if not spec_elems:
        spec_elems = root.findall(f".//specificity_row[@mod_key='{unimod_id}']")
    return list(spec_elems)


def _extract_modification_info(
    root: etree._Element, unimod_id: int
) -> Optional[ModificationInfo]:
    """Extract all information for a single UNIMOD modification.

    This includes the title, full name, monoisotopic mass, and specificities (including rare sites).
    """
    mod_elem = _find_modification_element(root, unimod_id)
    if mod_elem is None:
        print(f"  - WARNING: Could not find UNIMOD:{unimod_id}")
        return None

    title = _extract_required_attr(mod_elem, "code_name", unimod_id)
    full_name = _extract_required_attr(mod_elem, "full_name", unimod_id)
    mono_mass_str = _extract_required_attr(mod_elem, "mono_mass", unimod_id)

    try:
        mono_mass = float(mono_mass_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid mono_mass value '{mono_mass_str}' for UNIMOD:{unimod_id}"
        ) from e

    # Parse specificities
    all_specificities = []
    specificities = []

    for spec_elem in _find_specificity_elements(root, unimod_id):
        spec_info = _parse_specificity_element(spec_elem, unimod_id)
        all_specificities.append(spec_info)
        if not spec_info["rare"]:
            specificities.append(spec_info)

    print(f"  - UNIMOD:{unimod_id} ({title}): {mono_mass:.6f} Da")

    return {
        "id": unimod_id,
        "title": title,
        "full_name": full_name,
        "mono_mass": mono_mass,
        "specificities": specificities,
        "all_specificities": all_specificities,
    }


def query_unimod_database(unimod_ids: List[int]) -> Dict[int, ModificationInfo]:
    """Query UNIMOD database for each modification by parsing XML directly.

    This function finds the modification information for each UNIMOD ID in the database.
    Includes the title, full name, monoisotopic mass, and specificities (including rare sites).

    Returns:
        Dict mapping UNIMOD ID -> modification info
    """
    print("\nQuerying UNIMOD database...")

    cache_file = _get_unimod_cache()
    if cache_file is None:
        return {}

    root = _parse_unimod_xml(cache_file)
    if root is None:
        return {}

    mod_info: Dict[int, ModificationInfo] = {}

    for unimod_id in sorted(unimod_ids):
        try:
            info = _extract_modification_info(root, unimod_id)
            if info is not None:
                mod_info[unimod_id] = info
        except Exception as e:
            print(f"  - WARNING: Error processing UNIMOD:{unimod_id} - {e}")

    return mod_info


def _get_all_unimod_sites(mod: ModificationInfo) -> Set[str]:
    """Extract all amino acid sites (including rare) from a modification's specificities."""
    return {
        spec["site"]
        for spec in mod.get("all_specificities", mod["specificities"])
        if len(spec["site"]) == 1
    }


def validate_observed_sites(
    observed_combos: Dict[int, Set[str]], unimod_info: Dict[int, ModificationInfo]
) -> List[Tuple[int, str, str]]:
    """Check observed AA + modification combos against UNIMOD specificity data.

    Returns:
        List of (unimod_id, amino_acid, mod_title) tuples for combinations where
        the amino acid is not listed as a valid site in UNIMOD (potential mis-labelling).
    """
    print("\nValidating observed sites against UNIMOD specificities...")

    suspect: List[Tuple[int, str, str]] = []

    for unimod_id, observed_aas in sorted(observed_combos.items()):
        if unimod_id not in unimod_info:
            continue

        mod = unimod_info[unimod_id]
        all_sites = _get_all_unimod_sites(mod)
        observed_regular_aas = {a for a in observed_aas if a}

        unexpected = observed_regular_aas - all_sites
        for aa in sorted(unexpected):
            suspect.append((unimod_id, aa, mod["title"]))
            print(
                f"  - WARNING: Potential mis-labelling: {aa}[UNIMOD:{unimod_id}] "
                f"({mod['title']}) — {aa} is not a known UNIMOD site for this "
                f"modification (known sites: {', '.join(sorted(all_sites))})"
            )

    if not suspect:
        print(
            "  - All observed amino acid + modification combinations are valid UNIMOD sites"
        )
    else:
        print(f"  - Found {len(suspect)} potential mis-labelling(s)")

    return suspect


def _resolve_aa_mass(aa: str, unimod_id: int) -> float:
    """Look up the standard mass for an amino acid, warning and defaulting to 0 if missing."""
    aa_mass = mass.std_aa_mass.get(aa)
    if aa_mass is None:
        print(
            f"  - WARNING: No standard mass for amino acid '{aa}' "
            f"in {aa}[UNIMOD:{unimod_id}], defaulting to 0"
        )
        return 0.0
    return float(aa_mass)


def _add_modified_tokens(
    token_masses: Dict[str, float],
    observed_combos: Dict[int, Set[str]],
    unimod_info: Dict[int, ModificationInfo],
) -> None:
    """Add modified amino acid tokens to the mass dictionary (mutates token_masses)."""
    for unimod_id, observed_aas in sorted(observed_combos.items()):
        if unimod_id not in unimod_info:
            print(f"  - WARNING: Skipping UNIMOD:{unimod_id} (not in database)")
            continue

        mod_mass = unimod_info[unimod_id]["mono_mass"]

        for aa in sorted(observed_aas):
            if aa == "":
                token_masses[f"[UNIMOD:{unimod_id}]"] = mod_mass
            else:
                aa_mass = _resolve_aa_mass(aa, unimod_id)
                token_masses[f"{aa}[UNIMOD:{unimod_id}]"] = aa_mass + mod_mass


def build_mass_dictionary(
    observed_combos: Dict[int, Set[str]], unimod_info: Dict[int, ModificationInfo]
) -> Dict[str, float]:
    """Build the token -> mass dictionary.

    Only includes amino acid + modification combinations that were observed.
    """
    print("\nBuilding mass dictionary...")

    token_masses: Dict[str, float] = {}

    # Add standard amino acids
    for aa, aa_mass in mass.std_aa_mass.items():
        if aa == "J":
            # We do not want to include J (I/L ambiguity marker) in the mass dictionary
            # Instead, we map I and L to a single token (I or L) during preprocessing
            continue
        token_masses[aa] = aa_mass

    _add_modified_tokens(token_masses, observed_combos, unimod_info)

    print(f"  - Total tokens: {len(token_masses)}")
    print("  - Standard amino acids: 20")
    print(f"  - Modified tokens: {len(token_masses) - 20}")

    return token_masses


def _get_unimod_id_from_token(token: str) -> int:
    """Extract UNIMOD ID from a token string."""
    match = re.search(r"UNIMOD:(\d+)", token)
    return int(match.group(1)) if match else 0


def generate_yaml_output(token_masses: Dict[str, float], output_path: Path) -> None:
    """Generate YAML file with residue masses."""
    print(f"\nGenerating YAML output: {output_path}")

    lines = ["residues:"]

    # Standard amino acids first
    lines.append("  # Standard amino acids")
    for aa in sorted(k for k in token_masses if len(k) == 1):
        lines.append(f'  "{aa}": {token_masses[aa]:.6f}')

    # Modified amino acids (not terminal modifications)
    lines.append("")
    lines.append("  # Modified amino acids")

    aa_modified_tokens = sorted(
        (k for k in token_masses if "UNIMOD" in k and not k.startswith("[")),
        key=lambda x: (_get_unimod_id_from_token(x), x),
    )

    for token in aa_modified_tokens:
        lines.append(f'  "{token}": {token_masses[token]:.6f}')

    # N-terminal modifications at the end
    terminal_tokens = sorted(
        (k for k in token_masses if k.startswith("[UNIMOD")),
        key=_get_unimod_id_from_token,
    )

    if terminal_tokens:
        lines.append("")
        lines.append("  # N-terminal modifications")
        for token in terminal_tokens:
            lines.append(f'  "{token}": {token_masses[token]:.6f}')

    output_path.write_text("\n".join(lines) + "\n")
    print(f"  - Wrote {len(lines)} lines")


def _generate_modification_section(
    unimod_id: int,
    mod: ModificationInfo,
    observed_aas: Set[str],
    token_masses: Dict[str, float],
) -> List[str]:
    """Generate markdown lines for a single modification."""
    lines = [
        f"### UNIMOD:{unimod_id} - {mod['title']}",
        "",
        f"- **Full name**: {mod['full_name']}",
        f"- **Mass delta**: {mod['mono_mass']:+.6f} Da",
        "",
    ]

    # Observed amino acids
    obs_list = _format_amino_acid_list(observed_aas)
    if "" in observed_aas:
        obs_list[0] = "N-terminal"  # Replace "N-term" with "N-terminal" for report
    lines.append(f"- **Observed in data**: {', '.join(obs_list)}")
    lines.append("")

    # Generated tokens
    lines.append("- **Generated tokens**:")
    lines.extend(_generate_token_lines(unimod_id, mod, observed_aas, token_masses))
    lines.append("")

    # Site information
    lines.extend(_generate_site_lines(mod, observed_aas))

    lines.extend(["", "---", ""])
    return lines


def _generate_token_lines(
    unimod_id: int,
    mod: ModificationInfo,
    observed_aas: Set[str],
    token_masses: Dict[str, float],
) -> List[str]:
    """Generate token breakdown lines for a modification."""
    lines = []
    for aa in sorted(observed_aas):
        if aa == "":
            token = f"[UNIMOD:{unimod_id}]"
            total_mass = token_masses[token]
            lines.append(
                f"  - `{token}`: {total_mass:.6f} Da (N-terminal modification)"
            )
        else:
            token = f"{aa}[UNIMOD:{unimod_id}]"
            aa_mass = mass.std_aa_mass.get(aa, 0)
            total_mass = token_masses[token]
            lines.append(
                f"  - `{token}`: {total_mass:.6f} Da "
                f"({aa}: {aa_mass:.6f} + {mod['mono_mass']:.6f})"
            )
    return lines


def _check_site_coverage(
    unimod_sites: Set[str], observed_sites: Set[str]
) -> Optional[str]:
    """Return a coverage note if observed amino acids are a subset of or equal to UNIMOD sites."""
    if not unimod_sites or not observed_sites:
        return None
    if observed_sites < unimod_sites:
        missing = unimod_sites - observed_sites
        return (
            f"- **Note**: Not all possible non-rare sites observed "
            f"(missing: {', '.join(sorted(missing))})"
        )
    if observed_sites == unimod_sites:
        return "- **Note**: All possible non-rare amino acid sites observed"
    return None


def _check_unexpected_sites(
    all_unimod_sites: Set[str], observed_sites: Set[str]
) -> Optional[str]:
    """Return a mis-labelling warning if observed amino acids are outside all UNIMOD sites."""
    unexpected = observed_sites - all_unimod_sites
    if not unexpected:
        return None
    return (
        f"- **⚠️ Potential mis-labelling**: Observed on "
        f"{', '.join(sorted(unexpected))}, which "
        f"{'is' if len(unexpected) == 1 else 'are'} not listed as "
        f"{'a ' if len(unexpected) == 1 else ''}valid UNIMOD "
        f"{'site' if len(unexpected) == 1 else 'sites'} "
        f"(not even as rare)"
    )


def _generate_site_lines(mod: ModificationInfo, observed_aas: Set[str]) -> List[str]:
    """Generate site information lines for a modification."""
    all_sites: Set[str] = set()
    non_rare_sites: Set[str] = set()

    for spec in mod.get("all_specificities", mod["specificities"]):
        site = spec["site"]
        all_sites.add(site)
        if not spec.get("rare", False):
            non_rare_sites.add(site)

    # Format sites list
    all_sites_list = [
        site if site in non_rare_sites else f"{site} (rare)"
        for site in sorted(all_sites)
    ]

    lines = [f"- **All possible sites from UNIMOD**: {', '.join(all_sites_list)}"]

    unimod_aas = {s for s in non_rare_sites if len(s) == 1}
    all_unimod_aas = {s for s in all_sites if len(s) == 1}
    observed_regular_aas = {a for a in observed_aas if a}

    coverage_note = _check_site_coverage(unimod_aas, observed_regular_aas)
    if coverage_note:
        lines.append(coverage_note)

    mislabel_warning = _check_unexpected_sites(all_unimod_aas, observed_regular_aas)
    if mislabel_warning:
        lines.append(mislabel_warning)

    return lines


def generate_markdown_report(
    observed_combos: Dict[int, Set[str]],
    unimod_info: Dict[int, ModificationInfo],
    token_masses: Dict[str, float],
    suspect_sites: List[Tuple[int, str, str]],
    output_path: Path,
) -> None:
    """Generate markdown validation report."""
    print(f"\nGenerating markdown report: {output_path}")

    lines = [
        "# UNIMOD Modification Validation Report",
        "",
        "This report documents all UNIMOD modifications discovered from the input data files.",
        "",
        "## Summary",
        "",
        f"- **Total unique UNIMOD IDs**: {len(observed_combos)}",
        f"- **Total tokens generated**: {len(token_masses)}",
        "- **Standard amino acids**: 20",
        f"- **Modified tokens**: {len(token_masses) - 20}",
        "",
    ]

    if suspect_sites:
        lines.extend(
            [
                f"- **⚠️ Potential mis-labellings**: {len(suspect_sites)}",
                "",
                "---",
                "",
                "## ⚠️ Potential Mis-labellings",
                "",
                "The following amino acid + modification combinations were observed in the "
                "data but are **not** listed as valid sites in the UNIMOD database (not even "
                "as rare). These may indicate labelling errors in the source Excel files.",
                "",
                "| Token | Modification | Known UNIMOD Sites |",
                "|-------|-------------|-------------------|",
            ]
        )
        for unimod_id, aa, title in suspect_sites:
            all_sites = _get_all_unimod_sites(unimod_info[unimod_id])
            lines.append(
                f"| `{aa}[UNIMOD:{unimod_id}]` | {title} "
                f"| {', '.join(sorted(all_sites))} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Modifications by UNIMOD ID",
            "",
        ]
    )

    for unimod_id in sorted(observed_combos.keys()):
        if unimod_id not in unimod_info:
            continue

        mod = unimod_info[unimod_id]
        observed_aas = observed_combos[unimod_id]
        lines.extend(
            _generate_modification_section(unimod_id, mod, observed_aas, token_masses)
        )

    output_path.write_text("\n".join(lines))
    print(f"  - Wrote {len(lines)} lines")


def export_custom_modifications(custom_df: pl.DataFrame, output_path: Path) -> None:
    """Export custom IN:xxx modifications to Excel for expert annotation."""
    print(f"\nExporting custom modifications: {output_path}")

    if len(custom_df) == 0:
        print("  - No custom modifications found")
        return

    summary = (
        custom_df.group_by(["modification", "proposed_unimod_encoding", "project_name"])
        .agg(
            [
                pl.col("file_name").first().alias("example_file_name"),
                pl.len().alias("observed_count"),
            ]
        )
        .sort("proposed_unimod_encoding", "modification")
    )

    summary = summary.rename({"proposed_unimod_encoding": "proposed_encoding"})

    column_order = [
        "modification",
        "proposed_encoding",
        "project_name",
        "example_file_name",
        "observed_count",
    ]
    summary = summary.select([c for c in column_order if c in summary.columns])

    summary.write_excel(output_path)

    print(f"  - Exported {len(summary)} unique custom modification combinations")
    print(f"  - Total occurrences: {summary['observed_count'].sum()}")


def run_pipeline(
    gold_standard_path: Path,
    ambiguous_mods_path: Path,
    output_dir: Path,
) -> None:
    """Run the full UNIMOD mass dictionary building pipeline."""
    print("=" * 80)
    print("Building UNIMOD Mass Dictionary")
    print("=" * 80)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    residue_masses_output = output_dir / "residue_masses.yaml"
    markdown_report_output = output_dir / "modification_validation_report.md"
    custom_mods_output = output_dir / "custom_modifications_for_annotation.xlsx"

    # Step 1: Read Excel files
    gold_standard, ambiguous_mods = read_excel_files(
        gold_standard_path, ambiguous_mods_path
    )

    # Step 2: Separate UNIMOD vs custom modifications
    unimod_df, custom_df = separate_modification_types(gold_standard, ambiguous_mods)

    # Step 3: Export custom modifications
    export_custom_modifications(custom_df, custom_mods_output)

    # Step 4: Extract observed combinations
    observed_combos = extract_observed_combinations(unimod_df)

    # Step 5: Query UNIMOD database
    unimod_info = query_unimod_database(list(observed_combos.keys()))

    # Step 6: Validate observed sites against UNIMOD specificities
    suspect_sites = validate_observed_sites(observed_combos, unimod_info)

    # Step 7: Build mass dictionary
    token_masses = build_mass_dictionary(observed_combos, unimod_info)

    # Step 8: Generate outputs
    generate_yaml_output(token_masses, residue_masses_output)
    generate_markdown_report(
        observed_combos,
        unimod_info,
        token_masses,
        suspect_sites,
        markdown_report_output,
    )

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"  1. {residue_masses_output.name}")
    print(f"  2. {markdown_report_output.name}")
    print(f"  3. {custom_mods_output.name}")
    print()


@app.command()
def main(
    gold_standard: Annotated[
        Path,
        typer.Option(
            "--gold-standard",
            "-g",
            help="Path to gold standard modifications Excel file.",
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_GOLD_STANDARD,
    ambiguous_mods: Annotated[
        Path,
        typer.Option(
            "--ambiguous-mods",
            "-a",
            help="Path to ambiguous modifications Excel file.",
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_AMBIGUOUS_MODS,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for generated files.",
            file_okay=False,
        ),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Build UNIMOD mass dictionary from Excel modification annotation files."""
    run_pipeline(gold_standard, ambiguous_mods, output_dir)


if __name__ == "__main__":
    app()
