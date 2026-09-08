"""Delete redundant IPC variants that occur within the same folder.

Run this after ``detect_all_duplicates.py`` has produced a candidate report.
Same-folder copies can be resolved deterministically, but dry-run should be used
to review associated Parquet deletions first.

CLI::

    python scripts/preprocessing/delete_same_folder_duplicates.py --help
    python scripts/preprocessing/delete_same_folder_duplicates.py --input-file duplicate_files_acfm.txt --dry-run
    python scripts/preprocessing/delete_same_folder_duplicates.py --input-file report_a.txt --input-file report_b.txt --force

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, List

import typer

app = typer.Typer(
    help="Delete same-folder duplicate files",
    no_args_is_help=True,
    add_completion=False,
)


def delete_files_safely(files_to_delete: list) -> int:
    """Continue an approved cleanup even when individual files cannot be removed.

    Args:
        files_to_delete: Paths approved for removal.

    Returns:
        Number of files removed successfully.
    """
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            typer.echo(f"Deleted: {file_path}")
            deleted_count += 1
        except FileNotFoundError:
            typer.echo(f"File not found (skipped): {file_path}", err=True)
        except PermissionError:
            typer.echo(f"Permission denied (skipped): {file_path}", err=True)
        except Exception as e:
            typer.echo(f"Error deleting {file_path}: {e}", err=True)

    return deleted_count


def check_delete_with_user(files_to_delete: list, force: bool = False) -> None:
    """Require review before applying an irreversible duplicate cleanup.

    Args:
        files_to_delete: Paths proposed for removal.
        force: Whether prior approval permits bypassing the prompt.
    """
    typer.echo(f"The following {len(files_to_delete)} files will be deleted:")
    for file_path in files_to_delete:
        typer.echo(f"  {file_path}")

    if force:
        typer.echo("Force mode enabled - proceeding with deletion")
        proceed = True
    else:
        proceed = typer.confirm("Do you want to proceed with deletion?")

    if proceed:
        deleted_count = delete_files_safely(files_to_delete)
        typer.echo(f"\nDeletion process completed. {deleted_count} files deleted.")
    else:
        typer.echo("\nDeletion aborted by user.")


def parse_input_file(input_file: str) -> defaultdict:
    """Reconstruct experiment groups needed to distinguish same-folder copies.

    Args:
        input_file: Duplicate report produced by the all-duplicates detector.

    Returns:
        Experiment basenames mapped to their folders and full paths.
    """
    file_map = defaultdict(list)

    with open(input_file, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                folder, file_name = os.path.split(line)
                base_name = file_name.rsplit(".", maxsplit=2)[0]
                file_map[base_name].append((folder, line))

    return file_map


def find_duplicates_to_delete(file_map: defaultdict) -> list:
    """Choose only redundant same-folder copies and their derived Parquet files.

    Args:
        file_map: Experiment groups from a duplicate report.

    Returns:
        Paths safe to propose for same-folder cleanup.
    """
    files_to_delete = []

    for _base_name, file_info in file_map.items():
        folder_groups = defaultdict(list)
        for folder, file_path in file_info:
            folder_groups[folder].append(file_path)

        for _folder, file_paths in folder_groups.items():
            if len(file_paths) > 1:
                file_paths.sort()
                second_file = file_paths[1]
                files_to_delete.append(second_file)
                parquet_file = second_file.rsplit(".", maxsplit=1)[0] + ".parquet"
                if os.path.exists(parquet_file):
                    files_to_delete.append(parquet_file)

    return files_to_delete


def delete_same_folder_duplicates(
    input_file: str, force: bool = False, dry_run: bool = False
) -> None:
    """Remove deterministic same-folder copies while preserving cross-folder cases.

    Args:
        input_file: Report produced by ``detect_all_duplicates.py``.
        force: Whether to bypass interactive confirmation.
        dry_run: Whether to preview without deleting files.

    Raises:
        typer.Exit: If the duplicate report does not exist.
    """
    if not os.path.exists(input_file):
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    file_map = parse_input_file(input_file)
    files_to_delete = find_duplicates_to_delete(file_map)

    if dry_run:
        typer.echo("DRY RUN - The following files would be deleted:")
        for file_path in files_to_delete:
            typer.echo(f"  {file_path}")
        typer.echo(f"Total: {len(files_to_delete)} files")
        return

    if files_to_delete:
        check_delete_with_user(files_to_delete, force=force)
    else:
        typer.echo("No same-folder duplicates found to delete.")


@app.command()
def main(
    input_file: Annotated[
        List[Path],
        typer.Option(
            "--input-file",
            help="Duplicate report from detect_all_duplicates (repeatable)",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Skip confirmation prompt"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Preview without deleting"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Apply a duplicate report only where copies share a parent folder."""
    if verbose:
        typer.echo(f"Input files: {input_file}")
        typer.echo(f"Force mode: {force}")
        typer.echo(f"Dry run: {dry_run}")

    for path in input_file:
        if not path.exists():
            typer.echo(
                f"Warning: Input file '{path}' does not exist, skipping...",
                err=True,
            )
            continue
        typer.echo(f"Processing: {path}")
        delete_same_folder_duplicates(str(path), force=force, dry_run=dry_run)


if __name__ == "__main__":
    app()
