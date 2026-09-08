# ruff: noqa: T201 - a CLI generator: the emitted LaTeX goes to stdout
r"""Emit the data-audit supplementary tables as LaTeX.

The manuscript has many supplementary tables; this generates only those that describe
the released corpus itself, so their numbers are produced by measurement rather than
transcribed:

    \label{stab:filter_attrition}     Quality-filter attrition per confidence tier
    \label{stab:charge_acquisition}   Precursor charge by acquisition mode
    \label{stab:modifications}        Modification vocabulary of the released corpus
    \label{stab:in_codes}             Unresolved modification codes in full

They are identified by label rather than by number, because the numbering shifts as
other supplementary tables are added or removed.

Inputs are the JSON the three audits write. Reading their files rather than their
console output matters: the first version of these tables was assembled from job logs,
which meant the per-token detail existed only until the pod was reclaimed.

    python scripts/release/audit/verify_filter_conformance.py --mount "$ROOT" --out-dir audit_out
    python scripts/release/audit/charge_acquisition_crosstab.py --mount "$ROOT" --out-dir audit_out
    python scripts/verification/inventory_modifications.py --mount "$ROOT" \
        --output-csv audit_out/modifications.csv
    python scripts/release/audit/check_unimod_coverage.py \
        --codes scripts/release/audit/in_code_masses.csv \
        --out audit_out/unimod_coverage.json
    python scripts/release/audit/make_data_audit_tables.py --audit-dir audit_out \
        --unimod-coverage audit_out/unimod_coverage.json --out data_audit_tables.tex

The emitted LaTeX needs `booktabs`, `longtable` and a `\mz{}` macro for $m/z$, all of
which the manuscript already defines. Numbers are written with `{,}` group separators so they
typeset correctly in math-free table cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TIERS = (("LCFM", "lcfm"), ("MCFM", "mcfm"), ("HCFM", "hcfm"))

# Config names are identifiers; captions need prose.
PROSE_NAME = {
    "lcfm_splits": "LCFM training",
    "mcfm_splits": "MCFM training",
    "hcfm_splits": "HCFM training",
    "lcfm_by_project": "unsplit LCFM",
    "mcfm_by_project": "unsplit MCFM",
    "hcfm_by_project": "unsplit HCFM",
}

CRITERIA = (
    (r"Retention time $>$ 10{,}800~s", "retention_time_gt_10800"),
    (r"Lower isolation offset $>$ 300~Da", "lower_offset_gt_300"),
    (r"Unresolved modification annotation", "glyco_sequences"),
    (r"Precursor charge $>$ 7", "charge_gt_7"),
    (r"Precursor \mz{} $>$ 2{,}000~Da", "precursor_mz_gt_2000"),
)


def tex_int(x: int) -> str:
    """Thousands separators that survive a LaTeX table cell."""
    return f"{x:,}".replace(",", "{,}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audit-dir", type=Path, required=True, help="directory holding the audit JSON")
    p.add_argument("--out", type=Path, required=True, help="LaTeX file to write")
    p.add_argument(
        "--unimod-coverage",
        type=Path,
        default=Path("audit_out/unimod_coverage.json"),
        help="JSON written by check_unimod_coverage.py",
    )
    p.add_argument(
        "--charge-config",
        default="lcfm_splits",
        help="config to tabulate in the charge/acquisition table",
    )
    return p.parse_args(argv)


def load(audit_dir: Path) -> tuple[dict, dict, dict]:
    """Load the three audit outputs, naming clearly whichever is missing."""
    wanted = {
        "conformance": audit_dir / "conformance.json",
        "crosstab": audit_dir / "charge_acquisition.json",
        "inventory": audit_dir / "modifications_summary.json",
    }
    missing = [str(p) for p in wanted.values() if not p.is_file()]
    if missing:
        raise SystemExit(
            "error: missing audit output(s):\n  "
            + "\n  ".join(missing)
            + "\nRun the three audits first; see this module's docstring."
        )
    return tuple(json.loads(p.read_text()) for p in wanted.values())  # type: ignore[return-value]


def table_attrition(conf: dict) -> list[str]:
    """S14: how many rows each retention criterion removes, per tier."""
    need = [f"{t}_by_project" for _, t in TIERS] + [f"{t}_splits" for _, t in TIERS]
    for k in need:
        if k not in conf:
            raise SystemExit(f"error: conformance.json has no entry for {k}")

    bp = {t: conf[f"{t}_by_project"] for _, t in TIERS}
    sp = {t: conf[f"{t}_splits"] for _, t in TIERS}
    # LaTeX-safe: overlap is the only derived number, and it is stated not assumed
    overlap = {
        t: sum(bp[t][k] for _, k in CRITERIA) - (bp[t]["rows"] - sp[t]["rows"]) for _, t in TIERS
    }
    # Phrased as a complete clause so subject/verb agreement holds however many tiers
    # have overlap, including none.
    parts = [
        f"{lbl} contains {tex_int(overlap[t])} PSMs that fail more than one criterion"
        for lbl, t in TIERS
        if overlap[t]
    ]
    multi = ", and ".join(parts) if parts else "no tier has a PSM failing more than one criterion"

    out = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Quality-filter attrition per confidence tier.}",
        r"PSMs removed from each unsplit tier by each retention criterion. Each PSM is counted",
        r"once for every criterion it fails, so a PSM failing two criteria contributes to both",
        r"counts. Two criteria remove nothing at all: no PSM in the corpus carries a precursor",
        r"charge above 7 or a precursor \mz{} above 2{,}000~Da, either before or after filtering.",
        r"The unresolved-modification criterion likewise removes nothing from MCFM or HCFM,",
        r"because those PSMs are already excluded during tier construction. The per-criterion",
        r"counts therefore sum exactly to the number removed for those two tiers, while",
        multi + r".}",
        r"\label{stab:filter_attrition}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Criterion & LCFM & MCFM & HCFM \\",
        r"\midrule",
        "Unsplit tier PSMs & " + " & ".join(tex_int(bp[t]["rows"]) for _, t in TIERS) + r" \\",
        r"\midrule",
    ]
    for label, key in CRITERIA:
        out.append(f"{label} & " + " & ".join(tex_int(bp[t][key]) for _, t in TIERS) + r" \\")
    out += [
        r"\midrule",
        "Retained (training partition) & "
        + " & ".join(tex_int(sp[t]["rows"]) for _, t in TIERS)
        + r" \\",
        "Retained fraction & "
        + " & ".join(f"{100.0 * sp[t]['rows'] / bp[t]['rows']:.2f}\\%" for _, t in TIERS)
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return out


def table_charge(xt: dict, config: str) -> list[str]:
    """S15: precursor charge against acquisition mode."""
    if config not in xt:
        raise SystemExit(f"error: charge_acquisition.json has no entry for {config}")
    cells = xt[config]["by_acquisition_charge"]
    modes = sorted({k.rsplit("|", 1)[0] for k in cells})
    charges = sorted({int(k.rsplit("|", 1)[1]) for k in cells})

    def cell(mode: str, charge: int) -> int:
        return int(cells.get(f"{mode}|{charge}", 0))

    totals = {m: sum(cell(m, c) for c in charges) for m in modes}
    grand = sum(totals.values())

    out = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Precursor charge by acquisition mode in the "
        + PROSE_NAME.get(config, config.replace("_", r"\_"))
        + r" partition.}",
        r"A precursor charge of 0 is recorded only for DIA spectra, where the precursor is an",
        r"isolation window rather than a selected ion and no single charge state is assigned.",
        r"Every spectrum is labelled DDA or DIA; the acquisition field is never absent.}",
        r"\label{stab:charge_acquisition}",
        r"\begin{tabular}{l" + "r" * (len(modes) + 1) + r"}",
        r"\toprule",
        "Precursor charge & " + " & ".join(modes) + r" & Total \\",
        r"\midrule",
    ]
    for c in charges:
        row = [cell(m, c) for m in modes]
        label = "null" if c < 0 else str(c)
        out.append(
            f"{label} & " + " & ".join(tex_int(v) for v in row) + f" & {tex_int(sum(row))} " + r"\\"
        )
    out += [
        r"\midrule",
        "Total & "
        + " & ".join(tex_int(totals[m]) for m in modes)
        + f" & {tex_int(grand)} "
        + r"\\",
        "Share of partition & "
        + " & ".join(f"{100.0 * totals[m] / grand:.1f}\\%" for m in modes)
        + r" & 100\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return out


def table_modifications(inv: dict) -> list[str]:
    """S16: the corpus's modification vocabulary."""
    kinds = inv["by_kind"]
    u, i, o = kinds["unimod"], kinds["in"], kinds["other"]
    rows, with_tok = inv["rows"], inv["rows_with_token"]
    out = [
        # [H] not [h]: this table must stay put, because the longtable that
        # follows it is not a float and would otherwise overtake it.
        r"\begin{table}[H]",
        r"\centering",
        # [H] is not a float, so setspace no longer single-spaces the caption for us.
        r"\singlespacing",
        r"\caption{\textbf{Modification vocabulary of the released corpus.}",
        r"Every bracketed modification token in the \texttt{sequence} column, across all six",
        r"released datasets. Standard amino acid residues are not counted; a token is counted",
        r"per residue it occurs on, so one modification seen on two residues counts twice.",
        r"\texttt{[UNIMOD:\textit{n}]} denotes a modification resolved against",
        r"UNIMOD and retained; \texttt{[IN:\textit{n}]} is an internal namespace for compositions",
        r"absent from UNIMOD, which the retention filter excludes. No token in any other notation",
        r"occurs anywhere, so every modification in the corpus is either resolved against UNIMOD",
        r"or explicitly marked as unresolved. Glycopeptides are not excluded as a class: those",
        r"whose glycan carries a UNIMOD identifier are retained.}",
        r"\label{stab:modifications}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Token class & Distinct modification tokens & Occurrences \\",
        r"\midrule",
        r"\texttt{[UNIMOD:\textit{n}]}, retained & "
        f"{tex_int(u['distinct'])} & {tex_int(u['occurrences'])} " + r"\\",
        r"\texttt{[IN:\textit{n}]}, excluded & "
        f"{tex_int(i['distinct'])} & {tex_int(i['occurrences'])} " + r"\\",
        f"Any other notation & {o['distinct']} & {o['occurrences']} " + r"\\",
        r"\midrule",
        "Total & "
        f"{tex_int(u['distinct'] + i['distinct'] + o['distinct'])} & "
        f"{tex_int(u['occurrences'] + i['occurrences'] + o['occurrences'])} " + r"\\",
        r"\midrule",
        r"Spectra examined & \multicolumn{2}{r}{" + tex_int(rows) + r"} \\",
        r"Spectra with $\geq$1 modification & \multicolumn{2}{r}{"
        + f"{tex_int(with_tok)}  ({100.0 * with_tok / rows:.1f}\\%)"
        + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out += [
        "",
        r"\vspace{0.6em}",
        r"\footnotesize Every \texttt{[IN:\textit{n}]} code, with its mass and the",
        r"result of an independent UNIMOD search, is listed in",
        r"Supplementary Table~\ref{stab:in_codes}.",
    ]
    out += [r"\end{table}", ""]
    return out


def table_in_codes(cov: dict) -> list[str]:
    """The unresolved-code listing: every [IN:n] code, its mass, and its UNIMOD check.

    Laid out as two side-by-side panels; a single column of this many rows would run to
    three pages without becoming easier to read.
    """
    codes = sorted(cov["codes"], key=lambda r: int(r["code"].split(":")[1]))
    far = sum(1 for r in codes if r["distance_da"] > 5.0)
    median = sorted(r["distance_da"] for r in codes)[len(codes) // 2]
    panel = (len(codes) + 1) // 2
    head = r"Code & \multicolumn{1}{c}{$\Delta$ mass} & Spectra & Nearest"

    cap = [
        r"\caption{\textbf{Unresolved modification codes in full.}",
        f"All {len(codes)} " + r"\texttt{[IN:\textit{n}]} codes observed in the corpus, the",
        r"notation used for a modification that does not resolve to a UNIMOD identifier.",
        r"Spectra carrying one are excluded by the retention filter, so this is the complete",
        r"vocabulary of what the quality filters drop on modification grounds.",
        r"$\Delta$ mass is the monoisotopic mass delta in Da. \emph{Nearest} is the distance",
        r"in Da to the closest UNIMOD record at the same residue, obtained by searching the",
        r"UNIMOD database distributed with pyOpenMS; a large distance is evidence that no",
        f"UNIMOD identifier exists for that composition. {far} of the {len(codes)} codes lie",
        f"further than 5~Da from any record, and the median distance is {median:.0f}~Da.",
        r"UNIMOD records are identified by elemental composition rather than by mass, and",
        r"isobaric compositions exist, so a close match is a candidate for re-annotation",
        r"rather than proof of identity.",
    ]
    cap.append(r"}")

    out = [
        r"\begingroup",
        # longtable is not a float, so it inherits neither of the two things setspace and
        # the class do for a table: single spacing, and a caption as wide as the text
        # block. Without both, the caption sets double-spaced inside LaTeX's 4in default
        # and comes out centred and narrow, unlike every other supplementary table.
        r"\singlespacing",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}lrrr@{\hskip 1.5em}|@{\hskip 1.5em}lrrr@{}}",
        *cap,
        r"\label{stab:in_codes} \\",
        r"\toprule",
        head + " & " + head + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        head + " & " + head + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for k in range(panel):
        cells = []
        for r in (codes[k], codes[k + panel] if k + panel < len(codes) else None):
            if r is None:
                cells.append(" &  &  & ")
                continue
            cells.append(
                r"\texttt{"
                + f"{r['site']}[{r['code']}]"
                + "}"
                + f" & {r['delta_mass_da']:.4f} & {tex_int(r['spectra'])}"
                + f" & {r['distance_da']:.4f}"
            )
        out.append(" & ".join(cells) + r" \\")
    out += [r"\end{longtable}", r"\endgroup", ""]
    return out


def main(argv: list[str] | None = None) -> int:
    """Write the four supplementary tables as LaTeX."""
    args = parse_args(argv)
    conf, xt, inv = load(args.audit_dir)
    lines: list[str] = [
        "% Generated by scripts/release/audit/make_data_audit_tables.py -- do not edit by hand.",
        f"% Source: {args.audit_dir}",
        "",
    ]
    lines += table_attrition(conf)
    lines += table_charge(xt, args.charge_config)
    lines += table_modifications(inv)
    lines += table_in_codes(json.loads(args.unimod_coverage.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    n_tab = sum(1 for x in lines if x.startswith(r"\begin{table}"))
    n_long = sum(1 for x in lines if x.startswith(r"\begin{longtable}"))
    print(f"wrote {args.out} ({n_tab} tables, {n_long} longtable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
