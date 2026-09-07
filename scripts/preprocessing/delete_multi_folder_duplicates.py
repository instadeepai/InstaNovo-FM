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
    """Delete files safely and return the count of successfully deleted files."""
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
    """Check the files to be deleted with the user before deletion."""
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
    """Parse the duplicates report to find files in the target folder."""
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
    """Deletes multi-folder duplicate files from 'target_folder'."""
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
    """Delete multi-folder duplicate files."""
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
    """Delete multi-folder duplicates from multiple input files."""
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
