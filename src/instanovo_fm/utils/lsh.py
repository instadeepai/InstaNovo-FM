"""Locality-sensitive hashing over peak lists.

Vendored from the internal repo ``instanovo/utils/lsh.py``. It is not exported
by the public ``instanovo`` package, and ``scripts/splitting/split_unlabelled_data.py``
needs it to group unlabelled ACFM spectra by approximate spectral similarity.
Kept byte-identical apart from this header so it can be diffed against the
original; if it is ever upstreamed, delete this copy and import from instanovo.
"""

import hashlib
import logging
from typing import Optional

import numpy as np
from numba import njit


# ── Spectrum binning (numba-accelerated) ─────────────────────────────────────


@njit()
def _bin_peak_list(peak_list: np.ndarray, max_mz: float, bin_step: float) -> np.ndarray:
    """Bin a single peak list into fixed-width m/z bins."""
    mzs, intensities = peak_list
    num_bins = int(max_mz / bin_step)
    binned = np.zeros(num_bins, dtype=np.float32)
    bin_indices = (mzs / bin_step).astype(np.int32)
    for i in range(len(mzs)):
        idx = bin_indices[i]
        if 0 <= idx < num_bins:
            binned[idx] += intensities[i]
    return binned


@njit()
def _bin_peak_lists(
    peak_lists: np.ndarray, max_mz: float, bin_step: float
) -> np.ndarray:
    """Bin a batch of peak lists."""
    num_bins = int(max_mz / bin_step)
    out = np.zeros((len(peak_lists), num_bins), dtype=np.float32)
    for i in range(len(peak_lists)):
        out[i] = _bin_peak_list(peak_lists[i], max_mz, bin_step)
    return out


def bin_peak_list(peak_list: np.ndarray, max_mz: float, bin_step: float) -> np.ndarray:
    """Public API: bin a single peak list."""
    return _bin_peak_list(peak_list, max_mz, bin_step)


def bin_peak_lists(
    peak_lists: np.ndarray, max_mz: float, bin_step: float
) -> np.ndarray:
    """Public API: bin a batch of peak lists."""
    return _bin_peak_lists(peak_lists, max_mz, bin_step)


# ── Hashing helpers ──────────────────────────────────────────────────────────


def _sha256_row(arr: np.ndarray) -> str:
    """SHA-256 hex digest of a single numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha256_rows(signs: np.ndarray) -> np.ndarray:
    """SHA-256 hash each row of a boolean matrix.

    A list comprehension is measurably faster than ``np.apply_along_axis``
    because the latter has significant per-call dispatch overhead.
    """
    return np.array(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in signs],
        dtype="S64",
    )


# ── Random projection ───────────────────────────────────────────────────────


class RandomProjection:
    """Random hyperplane projection for locality-sensitive hashing.

    Projects input vectors onto *n_hyperplanes* random directions and records
    the sign of each projection.  The sign vector is then SHA-256 hashed to a
    64-character hex string.
    """

    def __init__(self, n_elems: int, n_hyperplanes: int, seed: int = 3):
        # RandomState uses the same legacy Mersenne Twister algorithm as
        # np.random.seed() + np.random.randn(), producing an identical H matrix,
        # but without mutating global numpy random state.
        rs = np.random.RandomState(seed)
        self.H = rs.randn(n_hyperplanes, n_elems)

    def compute(
        self, x: np.ndarray, as_str: bool = True, batched: bool = False
    ) -> np.ndarray:
        """Compute random projections.

        Args:
            x: Input vector (1D) or batch of vectors (2D).
            as_str: Return SHA-256 hex strings (True) or raw booleans (False).
            batched: Treat *x* as a 2D batch of row vectors.
        """
        if batched:
            if x.ndim != 2:
                raise ValueError(f"x must be 2D for batched mode, got {x.ndim}D")
            signs = np.einsum("ij,kj->ki", self.H, x) >= 0
            return _sha256_rows(signs) if as_str else signs

        if x.ndim != 1:
            raise ValueError(f"x must be 1D for single mode, got {x.ndim}D")
        signs = self.H @ x >= 0
        return _sha256_row(signs) if as_str else signs


# ── Peak-list projections ────────────────────────────────────────────────────


class PeakListRandomProjection:
    """LSH for a single peak list: bin by m/z then project."""

    def __init__(
        self,
        bin_step: float = 1,
        max_mz: float = 1000.0,
        n_hyperplanes: int = 50,
        seed: int = 3,
    ):
        if max_mz % bin_step != 0:
            raise ValueError(
                f"max_mz ({max_mz}) must be evenly divisible by "
                f"bin_step ({bin_step})"
            )
        self.bin_step = bin_step
        self.max_mz = max_mz
        self.rand_projection = RandomProjection(
            int(max_mz / bin_step), n_hyperplanes, seed
        )

    def compute(self, peak_list: np.ndarray, as_str: bool = True) -> np.ndarray:
        """Compute LSH for a single peak list of shape ``(2, num_peaks)``."""
        if peak_list.ndim != 2:
            raise ValueError(f"peak_list must be 2D, got {peak_list.ndim}D")
        binned = _bin_peak_list(peak_list, self.max_mz, self.bin_step)
        return self.rand_projection.compute(binned, as_str=as_str)


class BatchedPeakListRandomProjection(PeakListRandomProjection):
    """Batched LSH for arrays of shape ``(N, 2, num_peaks)``.

    Processes spectra in sub-batches of *subbatch_size* to bound memory usage
    during the binning and projection steps.
    """

    def __init__(
        self,
        bin_step: float = 1,
        max_mz: float = 1000.0,
        n_hyperplanes: int = 50,
        subbatch_size: int = 32,
        seed: int = 3,
    ):
        super().__init__(bin_step, max_mz, n_hyperplanes, seed)
        self.subbatch_size = subbatch_size

    def compute(
        self,
        peak_lists: np.ndarray,
        as_str: bool = True,
        logger: Optional[logging.Logger] = None,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """Compute LSH hashes for a batch of peak lists."""
        if peak_lists.ndim != 3:
            raise ValueError(f"peak_lists must be 3D, got {peak_lists.ndim}D")

        n = peak_lists.shape[0]

        if not self.subbatch_size or self.subbatch_size >= n:
            return self._compute_batch(peak_lists, as_str)

        from tqdm import tqdm

        chunks: list[np.ndarray] = []
        with tqdm(total=n, disable=not progress_bar, desc="Computing LSHs") as pbar:
            for i in range(0, n, self.subbatch_size):
                end = min(i + self.subbatch_size, n)
                if logger:
                    logger.info(f"Computing LSH [{i}:{end}] / {n} ...")
                chunks.append(self._compute_batch(peak_lists[i:end], as_str))
                pbar.update(end - i)

        return np.concatenate(chunks)

    def _compute_batch(self, peak_lists: np.ndarray, as_str: bool) -> np.ndarray:
        """Bin and project one sub-batch."""
        binned = _bin_peak_lists(peak_lists, self.max_mz, self.bin_step)
        return self.rand_projection.compute(binned, as_str=as_str, batched=True)
