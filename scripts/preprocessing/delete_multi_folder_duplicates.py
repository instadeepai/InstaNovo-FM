"""Delete selected cross-folder duplicates after their ownership is resolved.

Run this only after reviewing a report from
``detect_multi_folder_duplicates.py``; use dry-run when validating the selected
target folder, especially for batch deletion.

CLI::

    python scripts/preprocessing/delete_multi_folder_duplicates.py --help
    python scripts/preprocessing/delete_multi_folder_duplicates.py delete-duplicates multi_folder_duplicates.txt <target-folder> --dry-run
    python scripts/preprocessing/delete_multi_folder_duplicates.py batch-delete report_a.txt report_b.txt <target-folder> --dry-run

Use ``python script.py command --help`` for flags.
"""

import os
from pathlib import Path
import typer

app = typer.Typer(help="Delete multi-folder duplicate files")

# Module-level constants to avoid B008 errors
FORCE_OPTION = typer.Option(False, "--force", "-f", help="Skip confirmation prompt")
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-d",
    help="Show what would be deleted without actually deleting",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
INPUT_FILE_ARG = typer.Argument(
    ..., help="Input file containing multi-folder duplicate information"
)
TARGET_FOLDER_ARG = typer.Argument(
    ..., help="Target folder from which to delete duplicates"
)
INPUT_FILES_ARG = typer.Argument(
    ..., help="Input files containing multi-folder duplicate information"
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
    # Safety check: Print files to be deleted
    typer.echo(f"The following {len(files_to_delete)} files will be deleted:")
    for file_path in files_to_delete:
        typer.echo(f"  {file_path}")

    if force:
        typer.echo("Force mode enabled - proceeding with deletion")
        proceed = True
    else:
        # Ask for user confirmation
        proceed = typer.confirm("Do you want to proceed with deletion?")

    if proceed:
        # Delete the files
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
def delete_duplicates(
    input_file: str = INPUT_FILE_ARG,
    target_folder: str = TARGET_FOLDER_ARG,
    force: bool = FORCE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Apply a reviewed cross-folder deletion report with safety controls.

    Args:
        input_file: Cross-folder duplicate report to apply.
        target_folder: Folder whose copies should be removed.
        force: Whether to bypass interactive confirmation.
        dry_run: Whether to preview without deleting files.
        verbose: Whether to print selected settings.
    """
    if verbose:
        typer.echo(f"Input file: {input_file}")
        typer.echo(f"Target folder: {target_folder}")
        typer.echo(f"Force mode: {force}")
        typer.echo(f"Dry run: {dry_run}")

    delete_multi_folder_duplicates(
        input_file, target_folder, force=force, dry_run=dry_run
    )


@app.command()
def batch_delete(
    input_files: list[str] = INPUT_FILES_ARG,
    target_folder: str = TARGET_FOLDER_ARG,
    force: bool = FORCE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Apply several reviewed reports to the same target folder.

    Args:
        input_files: Cross-folder duplicate reports to apply.
        target_folder: Folder whose copies should be removed.
        force: Whether to bypass interactive confirmation.
        dry_run: Whether to preview without deleting files.
    """
    for input_file in input_files:
        if not Path(input_file).exists():
            typer.echo(
                f"Warning: Input file '{input_file}' does not exist, skipping...",
                err=True,
            )
            continue

        typer.echo(f"Processing: {input_file}")
        delete_multi_folder_duplicates(
            input_file, target_folder, force=force, dry_run=dry_run
        )



if __name__ == "__main__":
    app()
