"""Delete IPC and Parquet variants listed by a preprocessing validation report.

Run this after reviewing a file list such as the output from
``find_empty_files.py``; dry-run is available to verify the expanded paths
before deletion.

CLI::

    python scripts/preprocessing/delete_files.py --help
    python scripts/preprocessing/delete_files.py delete output_files/small_files_hcfm.txt --dry-run
    python scripts/preprocessing/delete_files.py batch-delete report_a.txt report_b.txt --dry-run

Use ``python script.py command --help`` for flags.
"""

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
    """Remove both stored variants of reviewed bad inputs while retaining an audit log.

    Args:
        file_list_path: Text file containing paths selected for cleanup.
        error_log_path: Destination for missing paths and deletion failures.
        dry_run: Whether to preview without deleting files.

    Raises:
        typer.Exit: If the input list does not exist.
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
    error_log = Path(error_log_path)
    error_log.parent.mkdir(parents=True, exist_ok=True)

    with open(error_log, "w") as error_log_file:
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
                        error_log_file.write(error_message)
                else:
                    warning_message = f"File not found, skipping: {file}\n"
                    logger.warning(warning_message.strip())
                    error_log_file.write(warning_message)


@app.command()
def delete(
    file_list: str = FILE_LIST_ARG,
    error_log: str = ERROR_LOG_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Apply a reviewed cleanup list to its IPC and Parquet variants.

    Args:
        file_list: Text file containing paths selected for cleanup.
        error_log: Destination for skipped paths and errors.
        dry_run: Whether to preview without deleting files.
        verbose: Whether to print selected settings.
    """
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
    """Apply several reviewed cleanup lists in one run.

    Args:
        file_lists: Text files containing paths selected for cleanup.
        error_log: Shared destination for skipped paths and errors.
        dry_run: Whether to preview without deleting files.
    """
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
    """Preserve backwards-compatible cleanup of the historical hardcoded lists."""
    # Legacy behaviour for backwards compatibility
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
