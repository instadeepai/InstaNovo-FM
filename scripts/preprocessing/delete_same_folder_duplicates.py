import os
from collections import defaultdict
from typing import List
import typer

app = typer.Typer(help="Delete same-folder duplicate files")

# Module-level constants to avoid B008 errors
INPUT_FILE_ARG = typer.Argument(..., help="Input file containing duplicate information")
FORCE_OPTION = typer.Option(False, "--force", "-f", help="Skip confirmation prompt")
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-d",
    help="Show what would be deleted without actually deleting",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
INPUT_FILES_ARG = typer.Argument(
    ..., help="Input files containing duplicate information"
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


def parse_input_file(input_file: str) -> defaultdict:
    """Parse the input file to group files by their base name."""
    file_map = defaultdict(list)

    with open(input_file, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                # Remove the extension to group by base name
                folder, file_name = os.path.split(line)
                base_name = file_name.rsplit(".", maxsplit=2)[
                    0
                ]  # Remove .ipc or .mzML.ipc
                file_map[base_name].append((folder, line))

    return file_map


def find_duplicates_to_delete(file_map: defaultdict) -> list:
    """Find files to delete based on same-folder duplicates."""
    files_to_delete = []

    for _base_name, file_info in file_map.items():
        # Group files by their folder names
        folder_groups = defaultdict(list)
        for folder, file_path in file_info:
            folder_groups[folder].append(file_path)

        # Process only duplicates in the same folder
        for _folder, file_paths in folder_groups.items():
            if len(file_paths) > 1:
                # Sort the paths to ensure consistent ordering
                file_paths.sort()
                # Remove the second occurrence and check for associated .parquet file
                second_file = file_paths[1]
                files_to_delete.append(second_file)
                parquet_file = second_file.rsplit(".", maxsplit=1)[0] + ".parquet"
                if os.path.exists(parquet_file):
                    files_to_delete.append(parquet_file)

    return files_to_delete


def delete_same_folder_duplicates(
    input_file: str, force: bool = False, dry_run: bool = False
) -> None:
    """Deletes same-folder duplicate files from an input file of detected duplicates.

    This function takes the output file of 'detect_same_duplicates` as its input file.
    It will skip duplicates that are found in different subfolders.

    Parameters:
        input_file (str): The file where same-folder duplicates will be read from.
        force (bool): Skip confirmation prompt
        dry_run (bool): Show what would be deleted without actually deleting
    """
    if not os.path.exists(input_file):
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    # Parse the input file to group files by their base name
    file_map = parse_input_file(input_file)

    # Find files to delete
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
def delete_duplicates(
    input_file: str = INPUT_FILE_ARG,
    force: bool = FORCE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Delete same-folder duplicate files from detected duplicates."""
    if verbose:
        typer.echo(f"Processing input file: {input_file}")
        typer.echo(f"Force mode: {force}")
        typer.echo(f"Dry run: {dry_run}")

    delete_same_folder_duplicates(input_file, force=force, dry_run=dry_run)


@app.command()
def batch_delete(
    input_files: List[str] = INPUT_FILES_ARG,
    force: bool = FORCE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Delete duplicates from multiple input files."""
    for input_file in input_files:
        if not os.path.exists(input_file):
            typer.echo(
                f"Warning: Input file '{input_file}' does not exist, skipping...",
                err=True,
            )
            continue

        typer.echo(f"Processing: {input_file}")
        delete_same_folder_duplicates(input_file, force=force, dry_run=dry_run)


def main() -> None:
    """Entry point for the script to delete same-folder duplicate files."""
    # Legacy behavior for backward compatibility
    input_files = ["duplicate_files_acfm.txt", "duplicate_files_lcfm.txt"]

    for input_file in input_files:
        if os.path.exists(input_file):
            typer.echo(f"Processing: {input_file}")
            delete_same_folder_duplicates(input_file)
        else:
            typer.echo(f"Warning: Input file '{input_file}' does not exist", err=True)


if __name__ == "__main__":
    app()
