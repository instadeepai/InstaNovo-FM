r"""Check every ``[IN:n]`` modification code against UNIMOD via pyOpenMS.

The corpus annotates modifications as ``[UNIMOD:n]`` where the modification resolves
against UNIMOD, and as ``[IN:n]`` where it does not. Spectra carrying an ``[IN:n]``
code are excluded by the retention filter, so the claim "these have no UNIMOD
identifier" decides what is dropped from the release. This script tests that claim
independently, against the UNIMOD database that ships inside pyOpenMS.

For each code it searches UNIMOD for a record whose monoisotopic mass delta matches,
restricted to the residue the code is observed on, and reports the distance to the
nearest record.

**What this can and cannot establish.** UNIMOD records are identified by elemental
composition, not by mass, and distinct compositions can be isobaric. So a code with no
UNIMOD record near its mass is firm evidence that no UNIMOD identifier exists, while a
code that *does* have a near-isobaric record is only a candidate: confirming it would
need the annotated composition, which is not part of this repository. The script
therefore reports distances and flags candidates rather than asserting identity.

A control runs first. Mass tables of this kind mix two conventions -- some entries store
a residue's total mass (residue plus modification) and others store the modification
delta alone -- and reading a delta as a total, or the reverse, shifts every lookup by a
residue mass and silently turns real matches into apparent absences. The control takes
modifications whose UNIMOD identifier is already known, checks that each is recovered
under exactly one convention, and refuses to run if recovery is not total.

    python scripts/release/audit/check_unimod_coverage.py \
        --codes scripts/release/audit/in_code_masses.csv \
        --out audit_out/unimod_coverage.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Modifications whose UNIMOD identifier is known independently, used as the control.
# Each is (token residue, stored mass, expected UNIMOD record). The first group stores a
# total residue mass, the second a bare delta -- the mix the control exists to detect.
CONTROL_TOTAL = (("M", 147.0354, 35), ("C", 160.030649, 4), ("N", 115.026943, 7))
CONTROL_DELTA = (("N", 1216.4229, 137), ("N", 1435.5223 + 114.042927, 1481))

# Widest gap that still counts as isobaric. Glycan compositions in UNIMOD are separated
# by far more than this, so it discriminates without being generous.
ISOBARIC_DA = 0.02


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codes", type=Path, required=True, help="CSV of IN codes and masses")
    p.add_argument("--out", type=Path, required=True, help="where to write the JSON result")
    p.add_argument(
        "--isobaric-da",
        type=float,
        default=ISOBARIC_DA,
        help=f"treat a UNIMOD record within this many Da as isobaric (default {ISOBARIC_DA})",
    )
    return p.parse_args(argv)


def load_unimod() -> list[tuple[str, float, int, str]]:
    """Return every UNIMOD site-record pyOpenMS knows, as (residue, delta, id, name)."""
    import pyopenms as oms

    db = oms.ModificationsDB()
    recs = []
    for i in range(db.getNumberOfModifications()):
        r = db.getModification(i)
        record_id = r.getUniModRecordId()
        # getUniModRecordId() returns -1 for entries that are not UNIMOD records, and -1
        # is truthy -- testing the value alone silently indexes every modification.
        if record_id > 0:
            recs.append((r.getOrigin(), r.getDiffMonoMass(), record_id, r.getFullId()))
    return recs


def nearest(recs: list[tuple[str, float, int, str]], residue: str, delta: float) -> tuple:
    """Return the closest UNIMOD record on ``residue``, as (distance, id, name)."""
    at_site = [(abs(m - delta), rid, name) for res, m, rid, name in recs if res == residue]
    if not at_site:
        return (float("inf"), None, None)
    return min(at_site)


def run_control(recs: list[tuple[str, float, int, str]]) -> None:
    """Verify the lookup recovers known UNIMOD identifiers; exit if it does not."""
    import pyopenms as oms

    rdb = oms.ResidueDB()
    failures = []
    for residue, stored, expect in CONTROL_TOTAL:
        residue_mass = rdb.getResidue(residue).getMonoWeight(oms.Residue.ResidueType.Internal)
        dist, rid, _ = nearest(recs, residue, stored - residue_mass)
        if dist > ISOBARIC_DA or rid != expect:
            failures.append(f"total-mass control {residue} {stored} expected {expect}, got {rid}")
    for residue, stored, expect in CONTROL_DELTA:
        dist, rid, _ = nearest(recs, residue, stored)
        if dist > ISOBARIC_DA or rid != expect:
            failures.append(f"delta control {residue} {stored} expected {expect}, got {rid}")
    if failures:
        print("CONTROL FAILED - the UNIMOD lookup is not trustworthy:", file=sys.stderr)
        for f in failures:
            print(f"   {f}", file=sys.stderr)
        raise SystemExit(2)
    print(
        f"  control: {len(CONTROL_TOTAL) + len(CONTROL_DELTA)}/"
        f"{len(CONTROL_TOTAL) + len(CONTROL_DELTA)} known identifiers recovered"
    )


def main(argv: list[str] | None = None) -> int:
    """Check each IN code against UNIMOD and write the result as JSON."""
    args = parse_args(argv)
    recs = load_unimod()
    print(f"  UNIMOD site-records available in pyOpenMS: {len(recs)}")
    run_control(recs)

    results = []
    with args.codes.open() as fh:
        for row in csv.DictReader(fh):
            delta = float(row["delta_mass_da"])
            dist, rid, name = nearest(recs, row["site"], delta)
            results.append(
                {
                    "code": row["code"],
                    "site": row["site"],
                    "delta_mass_da": delta,
                    "spectra": int(row["spectra"]),
                    "nearest_unimod_id": rid,
                    "nearest_unimod_name": name,
                    "distance_da": round(dist, 4),
                    "isobaric_candidate": dist <= args.isobaric_da,
                }
            )

    candidates = [r for r in results if r["isobaric_candidate"]]
    payload = {
        "codes_checked": len(results),
        "isobaric_da": args.isobaric_da,
        "no_unimod_record": len(results) - len(candidates),
        "isobaric_candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload | {"codes": results}, indent=1))

    print(f"\n  codes checked                    : {len(results)}")
    print(f"  no UNIMOD record within {args.isobaric_da} Da : {payload['no_unimod_record']}")
    print(f"  isobaric candidates              : {len(candidates)}")
    for r in candidates:
        print(
            f"     {r['code']} ({r['site']}) {r['delta_mass_da']:.4f} Da, "
            f"{r['spectra']:,} spectra -> UniMod:{r['nearest_unimod_id']} "
            f"at {r['distance_da']:.4f} Da  {r['nearest_unimod_name']}"
        )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
