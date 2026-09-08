"""Shared ion visualization utilities for spectrum plotting.

Provides ion-type classification, color mappings, and annotation formatting
used by both spectrum_analyser.py and confidence_signal_analysis.py.
"""

from __future__ import annotations

import re


def categorize_ion(annotation: str) -> str:
    """Categorize ion annotation into type.

    Handles multiple annotation formats:
    - PyOpenMS: "b3+", "y5++", "b3-H2O+"
    - rustyms: "b3+", "y5++", "b3-H2O1+"
    - Conditional annotation: "b3+[+1]" (isotope), "b3+-H2O" (loss)
    - Precursor: "M+2H", "M+3H", "M+2H[+1]" (isotope)

    Args:
        annotation: Ion annotation string.

    Returns:
        Category string (e.g. "B-ion", "Y-ion (Loss)", "unannotated").
    """
    if not annotation:
        return "unannotated"

    # Check for precursor ions first (PSI mzPAF p^z or p-loss^z format)
    if annotation.startswith("p^") or annotation.startswith("p-"):
        if "[+" in annotation and "]" in annotation:
            return "Precursor (Isotope)"
        else:
            return "Precursor"

    # Check for isotope notation from conditional annotation
    if "[+" in annotation and "]" in annotation:
        base_ion = annotation.split("[")[0]
        if base_ion and base_ion[0].lower() in "byacxz":
            return base_ion[0].upper() + "-ion (Isotope)"
        return "Isotope"

    # Custom ions (diagnostic, reporter, immonium, glycan, phospho, etc.)
    if annotation.startswith("custom:"):
        parts = annotation.split(":")
        if len(parts) > 1:
            ion_name = parts[1].split("@")[0]
            if "glycan" in ion_name:
                return "Glycan"
            elif "phospho" in ion_name or "PO3" in ion_name:
                return "Phospho"
            elif "immonium" in ion_name:
                return "Immonium"
            elif "TMT" in ion_name or "iTRAQ" in ion_name:
                return "Reporter"
            elif "sulfate" in ion_name:
                return "Sulfate"
            else:
                return "Custom"

    # Handle neutral losses (multiple formats)
    has_loss = False
    if "-" in annotation:
        parts = annotation.split("-")
        if len(parts) == 2:
            loss_part = parts[1].split("+")[0].split("++")[0]
            if any(x in loss_part.upper() for x in ["H2O", "NH3", "CO", "H3PO4", "H3PO3"]):
                has_loss = True
                base_ion = parts[0]
                if base_ion and base_ion[0].lower() in "byacxz":
                    return base_ion[0].upper() + "-ion (Loss)"

    # Standard fragment ions (b, y, a, c, x, z)
    if annotation[0].lower() in "byacxz":
        ion_type = annotation[0].upper() + "-ion"
        if has_loss:
            return ion_type + " (Loss)"
        return ion_type

    return "Other"


# Ion-type colours — explicit hex values so stem() and bar()
# render exactly the intended colour (no CN format-string parsing).
# Format: category -> (hex_colour, alpha)
CATEGORY_COLORS: dict[str, tuple[str, float]] = {
    "unannotated": ("#BDBDBD", 0.5),
    "B-ion": ("#1f77b4", 1.0),
    "Y-ion": ("#d62728", 1.0),
    "A-ion": ("#2ca02c", 0.9),
    "C-ion": ("#9467bd", 0.9),
    "X-ion": ("#8c564b", 0.9),
    "Z-ion": ("#e377c2", 0.9),
    "Precursor": ("#000000", 1.0),
    "Precursor (Isotope)": ("#4a4a4a", 0.8),
    "B-ion (Loss)": ("#1f77b4", 0.6),
    "Y-ion (Loss)": ("#d62728", 0.6),
    "A-ion (Loss)": ("#2ca02c", 0.6),
    "C-ion (Loss)": ("#9467bd", 0.6),
    "X-ion (Loss)": ("#8c564b", 0.6),
    "Z-ion (Loss)": ("#e377c2", 0.6),
    "Loss": ("#2ca02c", 0.6),
    "B-ion (Isotope)": ("#1f77b4", 0.5),
    "Y-ion (Isotope)": ("#d62728", 0.5),
    "A-ion (Isotope)": ("#2ca02c", 0.5),
    "C-ion (Isotope)": ("#9467bd", 0.5),
    "X-ion (Isotope)": ("#8c564b", 0.5),
    "Z-ion (Isotope)": ("#e377c2", 0.5),
    "Isotope": ("#7f7f7f", 0.5),
    "Immonium": ("#8c564b", 0.9),
    "Glycan": ("#e377c2", 0.9),
    "Phospho": ("#ff7f0e", 0.9),
    "Reporter": ("#17becf", 0.9),
    "Sulfate": ("#ff7f0e", 0.8),
    "Custom": ("#9467bd", 0.8),
    "Other": ("#bcbd22", 0.7),
}

# Darkened text colours for annotation labels (readable on white background)
TEXT_COLORS: dict[str, str] = {
    "unannotated": "olive",
    "B-ion": "darkblue", "B-ion (Loss)": "darkblue", "B-ion (Isotope)": "darkblue",
    "Y-ion": "darkred", "Y-ion (Loss)": "darkred", "Y-ion (Isotope)": "darkred",
    "A-ion": "darkgreen", "A-ion (Loss)": "darkgreen", "A-ion (Isotope)": "darkgreen",
    "C-ion": "darkviolet", "C-ion (Loss)": "darkviolet", "C-ion (Isotope)": "darkviolet",
    "X-ion": "saddlebrown", "X-ion (Loss)": "saddlebrown", "X-ion (Isotope)": "saddlebrown",
    "Z-ion": "deeppink", "Z-ion (Loss)": "deeppink", "Z-ion (Isotope)": "deeppink",
    "Precursor": "black", "Precursor (Isotope)": "#4a4a4a",
    "Loss": "darkgreen", "Isotope": "dimgray",
    "Immonium": "saddlebrown", "Glycan": "deeppink",
    "Phospho": "darkorange", "Sulfate": "darkorange",
    "Reporter": "darkcyan", "Custom": "darkviolet", "Other": "olive",
}


def format_annotation_display(annotation: str) -> str:
    """Format an ion annotation for plot display.

    Shortens custom ion labels and cleans up rustyms-style annotations
    (e.g. trailing stoichiometry numbers on neutral losses).

    Args:
        annotation: Raw ion annotation string.

    Returns:
        Shortened display string.
    """
    if not annotation:
        return ""

    display = annotation

    if annotation.startswith("custom:"):
        parts = annotation.split(":")
        if len(parts) > 1:
            ion_name = parts[1].split("@")[0]
            if "_" in ion_name:
                display = ion_name.split("_", 1)[1]
            else:
                display = ion_name
    else:
        # Clean rustyms-style annotations: "b3-H2O1+" -> "b3-H2O+"
        display = re.sub(r'([A-Z]+)(\d+)(\+)', r'\1\3', display)
        display = re.sub(r'([A-Z]+)(\d+)(\+\+)', r'\1\3', display)
        display = re.sub(r'([A-Z]+)(\d+)$', r'\1', display)

    return display
