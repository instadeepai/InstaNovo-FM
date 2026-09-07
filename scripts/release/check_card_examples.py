# ruff: noqa: T201, S102 - a CLI check: it prints its findings, and running the card's
# own code is the entire point (exec of a file we author, not of untrusted input)
r"""Run the code examples out of the dataset card and check what they produce.

A card whose examples do not run is worse than a card without examples: the reader
concludes the dataset is broken rather than the documentation. And these examples are
the only place the join key is demonstrated, which is the one thing a reader is most
likely to get wrong.

The snippets are **extracted from the card and executed**, not copied here. A copy
would drift the first time the card is edited, and then this would verify something
nobody reads.

Selection is by substring, because not every example should run: the "Loading a specific
tier" section names lcfm deliberately, and running that would move ~900 GB to prove
syntax.

    python scripts/release/check_card_examples.py \
        --card scripts/release/dataset_card.md --contains registry_key
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from instanovo_fm.utils.hf_token import resolve_hf_token
except ImportError:  # pragma: no cover - the slim runner image has no instanovo_fm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_token import resolve_hf_token  # type: ignore[no-redef]

FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# What the join example must produce: every row of the tier, each carrying a split.
EXPECTED_JOINED_ROWS = 3_684_448


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--card", type=Path, required=True)
    p.add_argument(
        "--contains",
        default="registry_key",
        help="run only snippets containing this substring",
    )
    p.add_argument("--list", action="store_true", help="list the snippets and exit")
    return p.parse_args(argv)


def snippets(card: Path) -> list[str]:
    """Every ```python block in the card, in order."""
    return [m.group(1) for m in FENCE.finditer(card.read_text())]


def main(argv: list[str] | None = None) -> int:
    """Execute the selected snippets and check their results."""
    args = parse_args(argv)
    blocks = snippets(args.card)
    print(f"  card holds {len(blocks)} python snippet(s)")
    if args.list:
        for i, b in enumerate(blocks):
            print(f"\n--- snippet {i} ---\n{b}")
        return 0

    chosen = [(i, b) for i, b in enumerate(blocks) if args.contains in b]
    if not chosen:
        raise SystemExit(f"error: no snippet contains {args.contains!r}")

    token, var = resolve_hf_token()
    # The card's examples pass no token, exactly as a reader's would. While the repo is
    # private that only works with an ambient credential, so supply one; after the repo
    # goes public this line stops mattering and the snippets are unchanged either way.
    os.environ["HF_TOKEN"] = token
    print(f"  token from {var} (exported as HF_TOKEN; the snippets pass none)")

    failures = 0
    for index, block in chosen:
        print(f"\n{'=' * 70}\n  running snippet {index} verbatim:\n{'=' * 70}")
        for line in block.strip().splitlines():
            print(f"    {line}")
        scope: dict = {}
        try:
            exec(compile(block, f"<card snippet {index}>", "exec"), scope)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - the verdict is what matters
            print(f"\n  FAILED: {type(exc).__name__}: {exc}")
            import traceback

            for line in traceback.format_exc().splitlines():
                print(f"    {line}")
            failures += 1
            continue

        print("\n  ran without error")
        joined = scope.get("labelled")
        if joined is None:
            print("  (no `labelled` produced; nothing further to check)")
            continue

        rows = len(joined)
        print(f"  labelled: {rows:,} rows x {len(joined.columns)} columns")
        print(f"  columns: {joined.columns}")
        if rows != EXPECTED_JOINED_ROWS:
            print(f"  ROW COUNT WRONG: expected {EXPECTED_JOINED_ROWS:,}")
            failures += 1
        else:
            print(f"  row count correct ({rows:,})")

        # A left join that matched nothing still "works" and returns every row with a
        # null split -- the exact silent failure the card warns about, so assert on it.
        if "split" in joined.columns:
            matched = int(joined["split"].is_not_null().sum())
            pct = 100.0 * matched / rows if rows else 0.0
            print(f"  rows with a split assigned: {matched:,} ({pct:.2f}%)")
            if matched == 0:
                print("  JOIN MATCHED NOTHING - the documented key does not work")
                failures += 1
            elif pct < 99.0:
                print(f"  join is incomplete: {rows - matched:,} rows unmatched")
                failures += 1
            else:
                counts = joined["split"].value_counts().sort("count", descending=True)
                print(f"  split distribution:\n{counts}")
        else:
            print("  WARNING: no `split` column, so the join produced nothing useful")
            failures += 1

    print(f"\n{'=' * 70}")
    print(f"  snippets run: {len(chosen)}, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
