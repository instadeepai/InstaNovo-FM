import logging
from pathlib import Path
import typer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Delete files from a list")

# Module-level constants to avoid B008 errors
FILE_LIST_ARG = typer.Argument(..., help="File containing list of files to delete")
ERROR_LOG_OPTION = typer.Option(
    "error_log.txt", "--error-log", "-e", help="Error log file"
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-d",
    help="Show what would be deleted without actually deleting",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
FILE_LISTS_ARG = typer.Argument(..., help="Files containing lists of files to delete")
BATCH_ERROR_LOG_OPTION = typer.Option(
    "batch_error_log.txt", "--error-log", "-e", help="Error log file"
)


def delete_files_from_list(
    file_list_path: str, error_log_path: str, dry_run: bool = False
) -> None:
    """Deletes files listed in a given text file from the local filesystem and logs errors to a file.

    Args:
        file_list_path (str): Path to the text file containing file paths to delete.
        error_log_path (str): Path to the error log file.
        dry_run (bool): If True, only show what would be deleted without actually deleting.
    """
    if not Path(file_list_path).exists():
        typer.echo(f"Error: File list '{file_list_path}' does not exist", err=True)
        raise typer.Exit(1)

    with open(file_list_path, "r") as f:
        files_to_delete = [line.strip() for line in f.readlines()]

    if dry_run:
        typer.echo("DRY RUN - The following files would be deleted:")
        for file_path in files_to_delete:
            typer.echo(f"  {file_path}")
        typer.echo(f"Total: {len(files_to_delete)} files")
        return

    # Ensure error log directory exists
    error_log_path = Path(error_log_path)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(error_log_path, "w") as error_log:
        for file_path in files_to_delete:
            base_path = file_path.rsplit(".", 1)[0]  # Remove file extension
            ipc_file = base_path + ".ipc"
            parquet_file = base_path + ".parquet"

            for file in [ipc_file, parquet_file]:
                file_path_obj = Path(file)
                if file_path_obj.exists():
                    try:
                        file_path_obj.unlink()
                        logger.info(f"Deleted file: {file}")
                    except Exception as e:
                        error_message = f"Error deleting file {file}: {e}\n"
                        logger.error(error_message.strip())
                        error_log.write(error_message)
                else:
                    warning_message = f"File not found, skipping: {file}\n"
                    logger.warning(warning_message.strip())
                    error_log.write(warning_message)


@app.command()
def delete(
    file_list: str = FILE_LIST_ARG,
    error_log: str = ERROR_LOG_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Delete files from a list."""
    if verbose:
        typer.echo(f"File list: {file_list}")
        typer.echo(f"Error log: {error_log}")
        typer.echo(f"Dry run: {dry_run}")

    delete_files_from_list(file_list, error_log, dry_run=dry_run)


@app.command()
def batch_delete(
    file_lists: list[str] = FILE_LISTS_ARG,
    error_log: str = BATCH_ERROR_LOG_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Delete files from multiple lists."""
    for file_list in file_lists:
        if not Path(file_list).exists():
            typer.echo(
                f"Warning: File list '{file_list}' does not exist, skipping...",
                err=True,
            )
            continue

        typer.echo(f"Processing file list: {file_list}")
        delete_files_from_list(file_list, error_log, dry_run=dry_run)


def main() -> None:
    """Entry point for deleting empty files."""
    # Legacy behavior for backward compatibility
    empty_files_lists = [
        "output_files/small_files_hcfm.txt",
        "output_files/small_files_mcfm.txt",
    ]
    error_log_path = "output_files/error_log.txt"

    for file_list in empty_files_lists:
        if Path(file_list).exists():
            typer.echo(f"Processing: {file_list}")
            delete_files_from_list(file_list, error_log_path)
        else:
            typer.echo(f"Warning: File list '{file_list}' does not exist", err=True)


if __name__ == "__main__":
    app()
