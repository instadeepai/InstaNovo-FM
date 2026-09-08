"""Delete selected cross-folder duplicates after their ownership is resolved.

Run this only after reviewing a report from
``detect_multi_folder_duplicates.py``; use dry-run when validating the selected
target folder.

CLI::

    python scripts/preprocessing/delete_multi_folder_duplicates.py --help
    python scripts/preprocessing/delete_multi_folder_duplicates.py --input-file multi_folder_duplicates.txt --target-dir <target-folder> --dry-run
    python scripts/preprocessing/delete_multi_folder_duplicates.py --input-file report_a.txt --input-file report_b.txt --target-dir <target-folder> --force

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, List

import typer

app = typer.Typer(
    help="Delete multi-folder duplicate files",
    no_args_is_help=True,
    add_completion=False,
)


def delete_files_safely(files_to_delete: list) -> int:
    """Continue a reviewed deletion set even when individual paths cannot be removed.

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
    """Require confirmation so an ambiguous duplicate is not removed accidentally.

    Args:
        files_to_delete: Paths proposed for removal.
        force: Whether prior approval allows bypassing the interactive prompt.
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


def parse_files_to_delete(input_file: str, target_folder: str) -> list:
    """Limit deletion to reviewed duplicates under the chosen target folder.

    Args:
        input_file: Cross-folder duplicate report to interpret.
        target_folder: Folder whose copies should be selected.

    Returns:
        Concrete IPC paths selected for deletion.

    Raises:
        typer.Exit: If the report does not exist.
    """
    files_to_delete = []
    if not Path(input_file).exists():
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    with open(input_file, "r") as infile:
        lines = infile.readlines()

    current_file = None
    for line in lines:
        line = line.strip()
        if line.startswith("File:"):
            current_file = line.split("File: ")[1]
        elif line.startswith("-"):
            folder = line.split("- ")[1]
            if target_folder in folder and current_file:
                files_to_delete.append(
                    os.path.join(folder, current_file) + ".mzML.ipc"
                )  # TODO: patch this suffix addition better
    return files_to_delete


def delete_multi_folder_duplicates(
    input_file: str, target_folder: str, force: bool = False, dry_run: bool = False
) -> None:
    """Remove only the reviewed duplicate copies assigned to one target folder.

    Args:
        input_file: Cross-folder duplicate report to apply.
        target_folder: Folder whose duplicate copies should be removed.
        force: Whether to bypass interactive confirmation.
        dry_run: Whether to preview without deleting files.
    """
    files_to_delete = parse_files_to_delete(input_file, target_folder)

    if dry_run:
        typer.echo("DRY RUN - The following files would be deleted:")
        for file_path in files_to_delete:
            typer.echo(f"  {file_path}")
        typer.echo(f"Total: {len(files_to_delete)} files")
        return

    if files_to_delete:
        check_delete_with_user(files_to_delete, force=force)
    else:
        typer.echo("No multi-folder duplicates found to delete.")


@app.command()
def main(
    input_file: Annotated[
        List[Path],
        typer.Option(
            "--input-file",
            help="Cross-folder duplicate report (repeatable)",
        ),
    ],
    target_dir: Annotated[
        Path,
        typer.Option(
            "--target-dir",
            help="Folder whose duplicate copies should be removed",
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
    """Apply a reviewed cross-folder deletion report with safety controls."""
    if verbose:
        typer.echo(f"Input files: {input_file}")
        typer.echo(f"Target dir: {target_dir}")
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
        delete_multi_folder_duplicates(
            str(path), str(target_dir), force=force, dry_run=dry_run
        )


if __name__ == "__main__":
    app()
