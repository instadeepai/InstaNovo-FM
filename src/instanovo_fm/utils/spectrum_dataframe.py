"""`SpectrumDataFrame` with the two methods the evaluation harness needs.

`instanovo` 1.2.2 ships `SpectrumDataFrame` without `collect_chunked` or
`filter_by_value`; both were added in the internal repository after that
release. Rather than vendor the whole 2,100-line `data_handler` module — which
would duplicate the dependency's most central class and let it drift — this
subclasses the released one and adds just those two methods.

Both are copied from the internal repository unchanged. They rely only on state
the released `__init__` already sets (`_is_native`, `_file_paths`,
`_filter_series_per_file`, `_shuffle`, `_add_source_file_column`) and on two
methods it already defines (`_reset_current_file`, `_update_file_indices`), so
nothing else has to be back-ported.

`load` and `from_polars` are classmethods that construct via `cls(...)`, so
they return this subclass and the extra methods are available on the result.

Import `SpectrumDataFrame` from here rather than from `instanovo.utils.data_handler`
anywhere the evaluation path needs those two methods. Drop this module once an
`instanovo` release includes them.
"""

from __future__ import annotations

import random

import polars as pl
from instanovo.utils.data_handler import SpectrumDataFrame as _ReleasedSpectrumDataFrame


class SpectrumDataFrame(_ReleasedSpectrumDataFrame):  # type: ignore[misc]
    """The released `SpectrumDataFrame` plus `filter_by_value` and `collect_chunked`."""

    def filter_by_value(self, column: str, value_set: set) -> None:
        """Filter rows by membership in a set of values for a given column.

        Significantly faster than ``filter_rows`` with a Python lambda because
        it uses Polars' native ``is_in`` expression (vectorised C++) instead of
        ``map_elements`` (row-by-row Python).

        Args:
            column: Column name to filter on.
            value_set: Set of allowed values.  Rows whose ``column`` value is
                not in this set are dropped.
        """
        values_list = list(value_set)
        if self._is_native:
            for fp in self._file_paths:
                df = pl.scan_parquet(fp).collect()
                if column not in df.columns:
                    # Column absent — keep existing filter unchanged
                    continue
                keep = df[column].is_in(values_list)
                self._filter_series_per_file[fp] &= keep
            self._reset_current_file()
            if not self._shuffle:
                self._update_file_indices()
        else:
            assert self.df is not None
            if column in self.df.columns:
                self.df = self.df.filter(pl.col(column).is_in(values_list))

    def collect_chunked(self, max_samples: int | None = None, seed: int | None = None) -> pl.DataFrame:
        """Load filtered data file-by-file, stopping once max_samples rows are collected.

        Reads one parquet file at a time and applies the per-file filter masks already
        set by filter_by_value / sample_subset, so peak memory is bounded to one file
        plus the accumulated target rows rather than all files at once.

        Args:
            max_samples: Maximum number of rows to collect. None collects everything.
            seed: Random seed used when truncating a file to the remaining quota.

        Returns:
            pl.DataFrame with up to max_samples rows.
        """
        if not self._is_native:
            assert self.df is not None
            if max_samples is not None and len(self.df) > max_samples:
                return self.df.sample(max_samples, seed=seed, shuffle=True)
            return self.df

        chunks: list[pl.DataFrame] = []
        n_collected = 0

        rng = random.Random(seed)
        file_paths = sorted(self._file_paths)
        rng.shuffle(file_paths)

        for fp in file_paths:
            if max_samples is not None and n_collected >= max_samples:
                break

            mask = self._filter_series_per_file[fp]
            if not mask.any():
                continue

            chunk = pl.scan_parquet(fp).filter(mask).collect()
            chunk = SpectrumDataFrame._ensure_experiment_name(chunk, fp, add_source=self._add_source_file_column, force_source=True)

            if max_samples is not None:
                remaining = max_samples - n_collected
                if len(chunk) > remaining:
                    chunk = chunk.sample(remaining, seed=seed, shuffle=True)

            chunks.append(chunk)
            n_collected += len(chunk)

        if not chunks:
            return pl.DataFrame()

        return pl.concat(chunks, how="diagonal_relaxed")


__all__ = ["SpectrumDataFrame"]
