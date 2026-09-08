"""Tests for :mod:`instanovo_fm.utils.lsh`."""

import numpy as np

from instanovo_fm.utils.lsh import BatchedPeakListRandomProjection


def _peak_lists(seed: int, count: int) -> np.ndarray:
    """Create deterministic peak lists for LSH tests."""
    rng = np.random.default_rng(seed)
    mz_values = np.sort(rng.uniform(100, 900, size=(count, 200)), axis=1)
    intensity_values = rng.uniform(0.1, 1.0, size=(count, 200))
    return np.stack([mz_values, intensity_values], axis=1).astype(np.float32)


def _projector(*, seed: int = 42, subbatch_size: int = 1024):
    """Create a consistently configured random-projection model."""
    return BatchedPeakListRandomProjection(
        max_mz=1000.0,
        bin_step=1.0,
        n_hyperplanes=30,
        subbatch_size=subbatch_size,
        seed=seed,
    )


def test_lsh_hash_is_reproducible() -> None:
    """Equal inputs and seeds should produce equal hashes."""
    peak_lists = _peak_lists(seed=123, count=5)

    hashes1 = _projector().compute(peak_lists, as_str=True, progress_bar=False)
    hashes2 = _projector().compute(peak_lists, as_str=True, progress_bar=False)

    assert np.array_equal(hashes1, hashes2)


def test_lsh_hash_differs_with_different_seed() -> None:
    """Different projection seeds should change at least some hashes."""
    peak_lists = _peak_lists(seed=123, count=5)

    hashes1 = _projector(seed=42).compute(peak_lists, as_str=True, progress_bar=False)
    hashes2 = _projector(seed=99).compute(peak_lists, as_str=True, progress_bar=False)

    assert not np.array_equal(hashes1, hashes2)


def test_batch_computation_matches_single_batch() -> None:
    """Subbatching should not change computed hashes."""
    peak_lists = _peak_lists(seed=456, count=10)

    batched = _projector(subbatch_size=2).compute(peak_lists, as_str=True, progress_bar=False)
    single_batch = _projector(subbatch_size=10_000).compute(
        peak_lists, as_str=True, progress_bar=False
    )

    assert np.array_equal(batched, single_batch)


def test_identical_spectra_produce_same_hash() -> None:
    """Identical spectra should produce the same LSH hash."""
    spectrum = _peak_lists(seed=789, count=1)[0]
    peak_lists = np.stack([spectrum, spectrum, spectrum])

    hashes = _projector().compute(peak_lists, as_str=True, progress_bar=False)

    assert len(set(hashes.flatten())) == 1


def test_different_spectra_produce_different_hashes() -> None:
    """Spectra in disjoint m/z ranges should produce different hashes."""
    rng = np.random.default_rng(123)
    low_mz = np.sort(rng.uniform(100, 300, size=200))
    high_mz = np.sort(rng.uniform(700, 900, size=200))
    low_intensity = rng.uniform(0.5, 1.0, size=200)
    high_intensity = rng.uniform(0.5, 1.0, size=200)
    peak_lists = np.stack(
        [
            np.stack([low_mz, low_intensity]),
            np.stack([high_mz, high_intensity]),
        ]
    ).astype(np.float32)

    hashes = _projector().compute(peak_lists, as_str=True, progress_bar=False)

    assert hashes[0] != hashes[1]
