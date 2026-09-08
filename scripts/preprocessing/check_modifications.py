"""Validate observed modification labels against supported translation mappings.

Run this after ``find_modifications.py`` and before
``label_modifications.py`` so missing mappings or project-specific overrides are
identified before Parquet sequences are rewritten.

CLI::

    python scripts/preprocessing/check_modifications.py --help
    python scripts/preprocessing/check_modifications.py check-mods modifications.xlsx gold_standard.xlsx pxd009449.xlsx
    python scripts/preprocessing/check_modifications.py batch-check-mods modifications_a.xlsx modifications_b.xlsx --gold-standard gold_standard.xlsx --pxd009449 pxd009449.xlsx

Use ``python script.py command --help`` for flags.
"""

import importlib.util
from pathlib import Path
from typing import List

import polars as pl
import typer

# Import builder functions from label_modifications
try:
    from label_modifications import (
        create_mod_dict,
        read_gold_standard_modifications,
        read_pxd009449_ambiguous_modifications,
    )
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "label_modifications", Path(__file__).parent / "label_modifications.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load label_modifications module")
    label_modifications = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_modifications)
    create_mod_dict = label_modifications.create_mod_dict
    read_gold_standard_modifications = (
        label_modifications.read_gold_standard_modifications
    )
    read_pxd009449_ambiguous_modifications = (
        label_modifications.read_pxd009449_ambiguous_modifications
    )

app = typer.Typer(help="Check if modifications from Excel file are in mod_dict")

# Module-level constants to avoid B008 errors
EXCEL_FILE_ARG = typer.Argument(..., help="Path to Excel file with modifications")
GOLD_STANDARD_ARG = typer.Argument(..., help="Gold standard modifications Excel file")
PXD009449_ARG = typer.Argument(..., help="PXD009449 ambiguous modifications Excel file")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
BATCH_EXCEL_FILES_ARG = typer.Argument(..., help="Excel files to check")
BATCH_GOLD_STANDARD_OPT = typer.Option(
    ..., "--gold-standard", help="Gold standard modifications Excel file"
)
BATCH_PXD009449_OPT = typer.Option(
    ..., "--pxd009449", help="PXD009449 ambiguous modifications Excel file"
)


def _load_modifications(excel_file: str) -> set:
    """Fail early when an inventory cannot provide the labels needed for validation."""
    try:
        df = pl.read_excel(excel_file)
    except Exception as e:
        typer.echo(f"Error reading Excel file: {e}", err=True)
        raise typer.Exit(1)

    if "modification" not in df.columns:
        typer.echo(
            f"Error: 'modification' column not found in Excel file. "
            f"Available columns: {df.columns}",
            err=True,
        )
        raise typer.Exit(1)

    if df.height == 0 or df["modification"].dtype == pl.Null:
        return set()

    mods = df["modification"].drop_nulls().unique().to_list()
    return {m for m in mods if m is not None}


def _report_overrides(
    override_modifications: set,
    gold_mod_dict: dict,
    pxd009449_mod_dict: dict,
) -> None:
    """Make project-specific overrides visible before mappings are applied."""
    typer.echo(
        f"\nPXD009449 OVERRIDE MODIFICATIONS ({len(override_modifications)}):"
    )
    typer.echo("   These modifications have special handling for PXD009449:")
    for mod in sorted(override_modifications):
        original_value = gold_mod_dict.get(mod, "N/A")
        override_value = pxd009449_mod_dict[mod]
        typer.echo(f"   - {mod}: {original_value} → {override_value} (for PXD009449)")


def _report_missing(missing_modifications: set) -> None:
    """Show unsupported labels so maintainers know which mappings to add."""
    typer.echo(f"\nMISSING MODIFICATIONS ({len(missing_modifications)}):")
    typer.echo("   These modifications are in the Excel file but NOT in mod_dict:")
    for mod in sorted(missing_modifications):
        typer.echo(f"   - {mod}")
    typer.echo(
        "\nACTION REQUIRED: Add these modifications to mod_dict "
        "in label_modifications.py"
    )


def _report_extra(extra_modifications: set) -> None:
    """Expose unused mappings when a verbose audit needs to detect stale entries."""
    typer.echo(f"\nEXTRA MODIFICATIONS ({len(extra_modifications)}):")
    typer.echo("   These modifications are in mod_dict but not in the Excel file:")
    for mod in sorted(extra_modifications)[:20]:
        typer.echo(f"   - {mod}")
    if len(extra_modifications) > 20:
        typer.echo(f"   ... and {len(extra_modifications) - 20} more")


def check_modifications(
    excel_file: str,
    gold_standard_file: str,
    pxd009449_file: str,
    verbose: bool = False,
) -> None:
    """Block sequence relabelling when observed modifications lack a supported mapping.

    Args:
        excel_file: Inventory generated by ``find_modifications.py``.
        gold_standard_file: Workbook containing default mappings.
        pxd009449_file: Workbook containing project-specific ambiguous mappings.
        verbose: Whether to report extra and duplicate mapping details.

    Raises:
        typer.Exit: If the inventory is invalid or any observed mapping is missing.
    """
    gold_standard_df = read_gold_standard_modifications(gold_standard_file)
    pxd009449_df = read_pxd009449_ambiguous_modifications(pxd009449_file)

    gold_mod_dict = create_mod_dict(gold_standard_df)
    pxd009449_mod_dict = create_mod_dict(pxd009449_df)
    merged_mod_dict = {**gold_mod_dict, **pxd009449_mod_dict}

    if verbose:
        typer.echo(f"Reading Excel file: {excel_file}")

    found_modifications = _load_modifications(excel_file)

    if verbose:
        typer.echo(
            f"Found {len(found_modifications)} unique modifications in Excel file"
        )

    mod_dict_keys = set(merged_mod_dict.keys())
    missing_modifications = found_modifications - mod_dict_keys
    extra_modifications = mod_dict_keys - found_modifications
    override_modifications = found_modifications & set(pxd009449_mod_dict.keys())

    typer.echo("\n" + "=" * 80)
    typer.echo("MODIFICATION CHECK RESULTS")
    typer.echo("=" * 80)
    typer.echo(f"\nTotal modifications found in Excel file: {len(found_modifications)}")
    typer.echo(
        f"Total modifications in mod_dict (including PXD009449 overrides): "
        f"{len(mod_dict_keys)}"
    )
    typer.echo(
        f"Modifications in Excel AND mod_dict: "
        f"{len(found_modifications & mod_dict_keys)}"
    )

    if override_modifications:
        _report_overrides(override_modifications, gold_mod_dict, pxd009449_mod_dict)

    if missing_modifications:
        _report_missing(missing_modifications)
    else:
        typer.echo(
            "\nSUCCESS: All modifications from Excel file are present in mod_dict!"
        )

    if extra_modifications and verbose:
        _report_extra(extra_modifications)

    typer.echo("\n" + "=" * 80)

    if missing_modifications:
        raise typer.Exit(1)


@app.command()
def check_mods(
    excel_file: str = EXCEL_FILE_ARG,
    gold_standard_file: str = GOLD_STANDARD_ARG,
    pxd009449_file: str = PXD009449_ARG,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Verify one modification inventory before relabelling its dataset.

    Args:
        excel_file: Inventory generated by ``find_modifications.py``.
        gold_standard_file: Workbook containing default mappings.
        pxd009449_file: Workbook containing project-specific mappings.
        verbose: Whether to print expanded audit details.
    """
    check_modifications(
        excel_file=excel_file,
        gold_standard_file=gold_standard_file,
        pxd009449_file=pxd009449_file,
        verbose=verbose,
    )


@app.command()
def batch_check_mods(
    excel_files: List[str] = BATCH_EXCEL_FILES_ARG,
    gold_standard_file: str = BATCH_GOLD_STANDARD_OPT,
    pxd009449_file: str = BATCH_PXD009449_OPT,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Audit several inventories against the same authoritative mapping workbooks.

    Args:
        excel_files: Modification inventories to validate.
        gold_standard_file: Workbook containing default mappings.
        pxd009449_file: Workbook containing project-specific mappings.
        verbose: Whether to print expanded audit details.

    Raises:
        typer.Exit: If any observed modification is unsupported.
    """
    gold_standard_df = read_gold_standard_modifications(gold_standard_file)
    pxd009449_df = read_pxd009449_ambiguous_modifications(pxd009449_file)

    gold_mod_dict = create_mod_dict(gold_standard_df)
    pxd009449_mod_dict = create_mod_dict(pxd009449_df)
    merged_mod_dict = {**gold_mod_dict, **pxd009449_mod_dict}

    all_missing = set()
    all_found = set()

    for excel_file in excel_files:
        typer.echo(f"\n{'=' * 80}")
        typer.echo(f"Checking: {excel_file}")
        typer.echo("=" * 80)

        try:
            df = pl.read_excel(excel_file)
            if "modification" not in df.columns:
                typer.echo(f"Skipping {excel_file}: 'modification' column not found")
                continue

            found_modifications = set(df["modification"].unique().to_list())
            all_found.update(found_modifications)

            mod_dict_keys = set(merged_mod_dict.keys())
            missing_modifications = found_modifications - mod_dict_keys
            all_missing.update(missing_modifications)

            if missing_modifications:
                typer.echo(
                    f"Found {len(missing_modifications)} missing modifications"
                )
            else:
                typer.echo("All modifications are present in mod_dict")

        except Exception as e:
            typer.echo(f"Error processing {excel_file}: {e}", err=True)

    typer.echo("\n" + "=" * 80)
    typer.echo("BATCH CHECK SUMMARY")
    typer.echo("=" * 80)
    typer.echo(f"\nTotal unique modifications across all files: {len(all_found)}")
    typer.echo(
        f"Total modifications in mod_dict (including PXD009449 overrides): {len(merged_mod_dict.keys())}"
    )

    if all_missing:
        typer.echo(f"\nTOTAL MISSING MODIFICATIONS ({len(all_missing)}):")
        for mod in sorted(all_missing):
            typer.echo(f"   - {mod}")
        typer.echo(
            "\nACTION REQUIRED: Add these modifications to mod_dict in label_modifications.py"
        )
        raise typer.Exit(1)
    else:
        typer.echo("\nSUCCESS: All modifications are present in mod_dict!")


if __name__ == "__main__":
    app()
