"""Delete IPC and Parquet variants listed by a preprocessing validation report.

Run this after reviewing a file list such as the output from
``find_empty_files.py``; dry-run is available to verify the expanded paths
before deletion.

CLI::

    uv run python -m scripts.preprocessing.delete_files --help
    uv run python -m scripts.preprocessing.delete_files --input-file output_files/small_files_hcfm.txt --dry-run
    uv run python -m scripts.preprocessing.delete_files --input-file report_a.txt --input-file report_b.txt --error-log errors.txt

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, List

import typer

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Delete files from a list",
    no_args_is_help=True,
    add_completion=False,
)


def expansion_targets(file_path: str) -> list[str]:
    """Expand a list entry to the IPC and Parquet paths a real delete would unlink."""
    base_path = file_path.rsplit(".", 1)[0]
    return [base_path + ".ipc", base_path + ".parquet"]


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
        files_to_delete = [line.strip() for line in f.readlines() if line.strip()]

    expanded = [
        target
        for file_path in files_to_delete
        for target in expansion_targets(file_path)
    ]

    if dry_run:
        logger.info("DRY RUN - The following files would be deleted:")
        for file_path in expanded:
            logger.info(f"  {file_path}")
        logger.info(f"Total: {len(expanded)} files")
        return

    error_log = Path(error_log_path)
    error_log.parent.mkdir(parents=True, exist_ok=True)

    with open(error_log, "a") as error_log_file:
        for file in expanded:
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
def main(
    input_file: Annotated[
        List[Path],
        typer.Option(
            "--input-file",
            help="Text file listing paths to delete (repeatable)",
        ),
    ],
    error_log: Annotated[
        Path,
        typer.Option("--error-log", help="Error log file"),
    ] = Path("error_log.txt"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Preview without deleting"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Apply reviewed cleanup lists to IPC and Parquet variants."""
    configure_script_logging(verbose=verbose)

    logger.debug(f"Input files: {input_file}")
    logger.debug(f"Error log: {error_log}")
    logger.debug(f"Dry run: {dry_run}")

    # Truncate shared error log once at the start of a real run.
    if not dry_run:
        error_log.parent.mkdir(parents=True, exist_ok=True)
        error_log.write_text("")

    for path in input_file:
        if not path.exists():
            logger.warning(f"File list '{path}' does not exist, skipping...")
            continue
        logger.info(f"Processing file list: {path}")
        delete_files_from_list(str(path), str(error_log), dry_run=dry_run)


if __name__ == "__main__":
    app()
