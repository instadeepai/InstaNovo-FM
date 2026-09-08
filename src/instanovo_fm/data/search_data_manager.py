"""Search data metadata manager for InstaNovo Foundation Model.

Loads metadata from CSV or Excel files and provides fast lookups by filepath.
Designed to be lightweight and easy to disable/enable.

Usage:
    manager = SearchDataManager("data/datasets/search_data.csv")
    manager.load()

    # Get metadata for a spectrum
    meta = manager.get_metadata(filepath="path/to/file.mzML.gz")

    # Check if loaded
    if manager.is_loaded:
        print(manager.summary())
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Regex pattern to match shard suffixes like _0001-0002
_SHARD_PATTERN = re.compile(r"_\d{4}-\d{4}$")

# Common MS file extensions to strip
_MS_EXTENSIONS = (
    ".mzML.gz", ".mzML", ".mzXML", ".raw",
    ".wiff", ".d", ".mgf", ".gz"
)


def _normalize_path(path: str) -> str:
    """Normalize path separators to forward slashes.

    Args:
        path: Raw file path string.

    Returns:
        Normalized path with forward slashes.
    """
    return path.replace("\\", "/")


def _extract_lookup_key_from_usi(usi: str) -> tuple[str, str]:
    """Extract parent folder and filename from a USI string.

    USI format: mzspec:PROJECT:filename[:scan:index[:peptide]]
    Maps to the same parent/filename key used by the filepath-based index.

    Returns ("unknown", "unknown") if the USI is malformed.
    """
    parts = usi.split(":")
    if len(parts) < 3:
        return "unknown", "unknown"
    project = parts[1].strip()
    filename = parts[2].strip()
    if not project or not filename:
        return "unknown", "unknown"
    # Strip MS file extensions from the filename component (same as filepath path)
    for ext in _MS_EXTENSIONS:
        if filename.lower().endswith(ext.lower()):
            filename = filename[:-len(ext)]
            break
    filename = filename.split(".")[0]
    if _SHARD_PATTERN.search(filename):
        filename = filename[:-10]
    return project, filename


def _extract_lookup_key(filepath: str) -> tuple[str, str]:
    """Extract parent folder and filename for lookup key.

    Extracts:
        - Parent folder: immediate parent directory name
        - Filename: basename without extensions and shard suffix

    Args:
        filepath: Full file path.

    Returns:
        Tuple of (parent_folder, filename_root).
        Returns ("unknown", "unknown") if path is invalid.
    """
    if not filepath:
        return "unknown", "unknown"

    # Normalize path separators
    normalized = _normalize_path(filepath)

    # Extract parent folder name
    parent_folder = Path(normalized).parent.name or "unknown"

    # Extract filename and remove extensions
    filename = Path(normalized).name

    # Strip MS file extensions (handle multi-part extensions like .mzML.gz)
    for ext in _MS_EXTENSIONS:
        if filename.lower().endswith(ext.lower()):
            filename = filename[:-len(ext)]
            break

    # Remove any remaining single extensions
    filename = filename.split(".")[0]

    # Remove shard suffix (e.g., _0001-0002)
    if _SHARD_PATTERN.search(filename):
        filename = filename[:-10]  # Remove last 10 chars (_XXXX-XXXX)

    return parent_folder, filename


class SearchDataManager:
    """Manager for search data metadata with fast lookups.

    Provides O(1) lookup of metadata by filepath using parent/filename keys.
    Preserves original column names from CSV/Excel file. Supports both CSV and Excel formats.

    Args:
        file_path: Path to search data file (.csv or .xlsx).
        filepath_column: Column name with file paths (default: "file path").
        spectrum_filepath_key: Key name in spectrum dict with file path (default: "filepath").
    """

    def __init__(
        self,
        file_path: str,
        filepath_column: str = "file path",
        spectrum_filepath_key: str = "filepath",
    ):
        """Initialize search data manager."""
        self.file_path = file_path
        self.filepath_column = filepath_column
        self.spectrum_filepath_key = spectrum_filepath_key

        # Storage
        self._df: Optional[pd.DataFrame] = None
        self._lookup: Dict[str, Dict[str, Any]] = {}
        self._unindexed_rows: int = 0
        self._is_loaded: bool = False

        # Cache for _extract_lookup_key results (filepaths repeat across batches)
        self._key_cache: Dict[str, str] = {}

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load CSV/Excel file and build lookup index.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If required filepath column is missing.
        """
        # Check file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Search data file not found: {self.file_path}"
            )


        # Load file (auto-detect CSV vs Excel)
        if self.file_path.lower().endswith('.csv'):
            self._df = pd.read_csv(self.file_path)
        else:
            # Excel format - requires openpyxl
            try:
                self._df = pd.read_excel(self.file_path)
            except ImportError as e:
                if "openpyxl" in str(e):
                    raise ImportError(
                        "openpyxl is required to read Excel files. "
                        "Install it with: pip install openpyxl or uv pip install openpyxl. "
                        "Alternatively, convert your file to CSV format."
                    ) from e
                raise

        # Validate required column exists
        if self.filepath_column not in self._df.columns:
            raise ValueError(
                f"Required column '{self.filepath_column}' not found in file. "
                f"Available columns: {list(self._df.columns)}"
            )

        # Build lookup index
        self._lookup = {}
        self._unindexed_rows = 0

        for idx, row in self._df.iterrows():
            filepath = row.get(self.filepath_column)

            # Skip invalid paths
            if not isinstance(filepath, str) or not filepath:
                self._unindexed_rows += 1
                continue

            # Extract lookup key
            parent, filename = _extract_lookup_key(filepath)
            if parent == "unknown" or filename == "unknown":
                self._unindexed_rows += 1
                continue

            # Create composite key
            key = f"{parent}/{filename}"

            # Store metadata as dict with original column names
            self._lookup[key] = {col: row[col] for col in self._df.columns}

        self._is_loaded = True

        logger.info(f"Search data: {len(self._lookup)} rows indexed from {self.file_path}")

    def get_metadata(self, filepath: str) -> Dict[str, Any]:
        """Get metadata for a given filepath.

        Args:
            filepath: File path to lookup.

        Returns:
            Dictionary with metadata from file (original column names).
            Returns empty dict if not found or not loaded.
        """
        if not self._is_loaded:
            logger.warning("SearchDataManager not loaded, returning empty metadata")
            return {}

        if not filepath:
            return {}

        # Extract lookup key — support both file paths and USIs
        if filepath.startswith("mzspec:"):
            parent, filename = _extract_lookup_key_from_usi(filepath)
        else:
            parent, filename = _extract_lookup_key(filepath)
        if parent == "unknown" or filename == "unknown":
            return {}

        key = f"{parent}/{filename}"

        # Return copy to prevent external modifications
        return dict(self._lookup.get(key, {}))

    def get_metadata_batch(
        self,
        filepaths: list[str],
        columns: Optional[list[str]] = None,
    ) -> list[Dict[str, Any]]:
        """Get metadata for multiple filepaths (optimized batch processing).

        Uses caching to avoid recomputing lookup keys for repeated filepaths.

        Args:
            filepaths: List of file paths to lookup.
            columns: Optional list of specific columns to return. If None, returns all columns.
                Use this to avoid copying unnecessary columns for performance.

        Returns:
            List of metadata dictionaries (empty dicts for not found).
        """
        if not self._is_loaded:
            logger.warning("SearchDataManager not loaded, returning empty metadata batch")
            return [{} for _ in filepaths]

        results = []
        for filepath in filepaths:
            if not filepath:
                results.append({})
                continue

            # Check cache first (filepaths often repeat across batches)
            key = self._key_cache.get(filepath)
            if key is None:
                # Cache miss - compute and store; support both file paths and USIs
                if filepath.startswith("mzspec:"):
                    parent, filename = _extract_lookup_key_from_usi(filepath)
                else:
                    parent, filename = _extract_lookup_key(filepath)
                if parent == "unknown" or filename == "unknown":
                    results.append({})
                    continue
                key = f"{parent}/{filename}"
                self._key_cache[filepath] = key

            # Get metadata from lookup
            full_metadata = self._lookup.get(key)
            if not full_metadata:
                results.append({})
                continue

            # Filter to requested columns if specified (performance optimization)
            if columns:
                results.append({col: full_metadata.get(col) for col in columns})
            else:
                # Return copy to prevent external modifications
                results.append(dict(full_metadata))

        return results

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data.

        Returns:
            Dictionary with loading statistics.
        """
        if not self._is_loaded:
            return {
                "loaded": False,
                "file_path": self.file_path,
            }

        return {
            "loaded": True,
            "file_path": self.file_path,
            "total_rows": len(self._df) if self._df is not None else 0,
            "indexed_rows": len(self._lookup),
            "unindexed_rows": self._unindexed_rows,
            "columns": list(self._df.columns) if self._df is not None else [],
        }

    def get_available_columns(self) -> list[str]:
        """Get list of available metadata columns.

        Returns:
            List of column names from data file.
            Empty list if not loaded.
        """
        if not self._is_loaded or self._df is None:
            return []
        return list(self._df.columns)


def create_search_data_manager(
    config: Dict[str, Any]
) -> Optional[SearchDataManager]:
    """Factory function to create SearchDataManager from config.

    Args:
        config: Configuration dictionary with keys:
            - use_search_data: bool, whether to enable search data
            - search_data_path: str, path to CSV/Excel file
            - search_data_filepath_column: str, column name in file (optional)
            - search_data_spectrum_key: str, key in spectrum dict (optional)

    Returns:
        Initialized SearchDataManager if enabled, None otherwise.
    """
    # Check if feature is enabled
    if not config.get("use_search_data", False):
        return None

    # Get required path
    excel_path = config.get("search_data_path")
    if not excel_path:
        logger.warning(
            "use_search_data=True but search_data_path not provided. "
            "Disabling search data integration."
        )
        return None

    # Create manager
    manager = SearchDataManager(
        file_path=excel_path,
        filepath_column=config.get(
            "search_data_filepath_column", "file path"
        ),
        spectrum_filepath_key=config.get(
            "search_data_spectrum_key", "filepath"
        ),
    )

    # Load data
    try:
        manager.load()
        return manager
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(f"Failed to load search data: {e}")
        logger.warning("Continuing without search data integration")
        return None


__all__ = ["SearchDataManager", "create_search_data_manager"]
