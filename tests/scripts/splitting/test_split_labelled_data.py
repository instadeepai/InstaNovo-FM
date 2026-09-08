"""Tests for split_labelled_data."""

import ast
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import polars as pl
import pytest

from scripts.splitting.split_labelled_data import (
    REGISTRY_FILENAME,
    Mode,
    assign_new_peptides,
    collect_unique_peptides,
    load_peptide_registry,
    process_directories,
    save_registry,
    verify_no_unseen_peptides,
)

class TestSplitLabelledData:
    """Test suite for split_labelled_data.py — registry extension and leakage."""

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Temporary directories for local registries and parquet fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.registry_dir = Path(self.test_dir) / "registry"
        self.output_dir = Path(self.test_dir) / "output"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(self.test_dir)

    @staticmethod
    def _empty_splits() -> dict[str, set[str]]:
        return {"train": set(), "test": set(), "valid": set()}

    @staticmethod
    def _write_registry(
        directory: Path, splits: dict[str, set[str]], *, use_hf_valid: bool = False
    ) -> Path:
        """Write peptide_registry.parquet; HF schema uses 'validation' not 'valid'."""
        peptides: list[str] = []
        labels: list[str] = []
        for split_name, seqs in splits.items():
            label = (
                "validation"
                if use_hf_valid and split_name == "valid"
                else split_name
            )
            for peptide in sorted(seqs):
                peptides.append(peptide)
                labels.append(label)
        path = directory / REGISTRY_FILENAME
        pl.DataFrame({"peptide": peptides, "split": labels}).write_parquet(path)
        return path

    @staticmethod
    def _minimal_labelled_parquet(
        path: Path,
        sequences: list[str],
        *,
        unmodified: list[str] | None = None,
    ) -> None:
        """Write a tiny parquet that passes quality filters (null filter cols pass)."""
        n = len(sequences)
        data: dict = {
            "index": list(range(n)),
            "scan": [str(i) for i in range(n)],
            "header": [f"h{i}" for i in range(n)],
            "sequence": sequences,
            "mz_array": [[100.0 + i, 200.0 + i] for i in range(n)],
            "intensity_array": [[1.0, 2.0] for _ in range(n)],
        }
        if unmodified is not None:
            data["unmodified_peptide"] = unmodified
        # Leave retention_time / precursor_* / lower_offset absent so schema
        # normalisation fills nulls and _nullable_filter lets rows through.
        pl.DataFrame(data).write_parquet(path)

    def test_collect_unique_peptides_normalises_i_to_l(self) -> None:
        """I→L and unmodified_peptide-from-sequence feed the registry key space."""
        parquet = self.data_dir / "sample.parquet"
        self._minimal_labelled_parquet(
            parquet,
            sequences=["PEPTIDE[UNIMOD:4]", "PEPTLDE", "AAAA"],
            unmodified=None,
        )
        peptides = collect_unique_peptides([str(parquet)])
        assert "PEPTLDE" in peptides
        assert "PEPTIDE" not in peptides
        assert "AAAA" in peptides

    def test_assign_preserves_existing_and_places_only_new(self) -> None:
        """Seeded registry peptides stay put; only absent peptides are assigned."""
        existing = {
            "train": {"SEEDTRAIN", "SEEDTRAIN2"},
            "test": {"SEEDTEST"},
            "valid": {"SEEDVALID"},
        }
        dataset = {
            "SEEDTRAIN",
            "SEEDTRAIN2",
            "SEEDTEST",
            "SEEDVALID",
            "NEW1",
            "NEW2",
            "NEW3",
            "NEW4",
            "NEW5",
        }
        before = {k: set(v) for k, v in existing.items()}
        updated, added = assign_new_peptides(dataset, existing)

        assert updated["train"] >= before["train"]
        assert updated["test"] >= before["test"]
        assert updated["valid"] >= before["valid"]
        assert "SEEDTRAIN" in updated["train"]
        assert "SEEDTEST" in updated["test"]
        assert "SEEDVALID" in updated["valid"]

        new = {"NEW1", "NEW2", "NEW3", "NEW4", "NEW5"}
        assigned_new = (
            (updated["train"] & new)
            | (updated["test"] & new)
            | (updated["valid"] & new)
        )
        assert assigned_new == new
        assert sum(added.values()) == len(new)

        for peptide in new:
            membership = sum(peptide in updated[s] for s in ("train", "test", "valid"))
            assert membership == 1, f"{peptide} in {membership} splits"

    def test_no_peptide_leakage(self) -> None:
        """No peptide appears in more than one split after incremental updates."""
        existing = {
            "train": {f"TRAIN{i}" for i in range(40)},
            "test": {f"TEST{i}" for i in range(5)},
            "valid": {f"VALID{i}" for i in range(5)},
        }
        dataset = (
            existing["train"]
            | existing["test"]
            | existing["valid"]
            | {f"NEW{i}" for i in range(50)}
        )
        updated, _ = assign_new_peptides(dataset, existing)
        assert updated["train"].isdisjoint(updated["test"])
        assert updated["train"].isdisjoint(updated["valid"])
        assert updated["test"].isdisjoint(updated["valid"])

    def test_assign_is_deterministic(self) -> None:
        """Same seed and inputs yield identical assignments."""
        existing_a = self._empty_splits()
        existing_b = self._empty_splits()
        peptides = {f"P{i:03d}" for i in range(100)}
        a, _ = assign_new_peptides(peptides, existing_a)
        b, _ = assign_new_peptides(peptides, existing_b)
        assert a["train"] == b["train"]
        assert a["test"] == b["test"]
        assert a["valid"] == b["valid"]

    def test_saturation_skips_overfull_split(self) -> None:
        """When train already meets its share of dataset_peptides, new go elsewhere."""
        # |dataset|=92; train target ≈ 73.6; train ∩ dataset = 80 → train saturated
        train_seed = {f"TS{i}" for i in range(80)}
        existing = {
            "train": set(train_seed),
            "test": {"TE0"},
            "valid": {"TV0"},
        }
        news = {f"NEW{i}" for i in range(10)}
        dataset = train_seed | {"TE0", "TV0"} | news
        updated, added = assign_new_peptides(dataset, existing)
        assert added["train"] == 0
        assert news.isdisjoint(updated["train"])
        assert news <= (updated["test"] | updated["valid"])

    def test_save_load_registry_roundtrip_maps_validation(self) -> None:
        """HF 'validation' label loads as local 'valid'; round-trip preserves sets."""
        splits = {
            "train": {"AAA", "BBB"},
            "test": {"CCC"},
            "valid": {"DDD", "EEE"},
        }
        self._write_registry(self.registry_dir, splits, use_hf_valid=True)
        _, loaded = load_peptide_registry(str(self.registry_dir))
        assert loaded["train"] == splits["train"]
        assert loaded["test"] == splits["test"]
        assert loaded["valid"] == splits["valid"]

        out = self.output_dir / REGISTRY_FILENAME
        save_registry(loaded, out, upload_to_hf=False)
        _, reloaded = load_peptide_registry(str(self.output_dir))
        assert reloaded == loaded

    def test_empty_registry_edge_case_fills_ratios(self) -> None:
        """0-row stub registry → empty sets; assign_new_peptides fills ~80/10/10."""
        self._write_registry(self.registry_dir, self._empty_splits())
        _, existing = load_peptide_registry(str(self.registry_dir))
        assert existing == self._empty_splits()

        peptides = {f"P{i:03d}" for i in range(100)}
        updated, added = assign_new_peptides(peptides, existing)
        total = sum(len(updated[s]) for s in ("train", "test", "valid"))
        assert total == 100
        assert sum(added.values()) == 100
        assert updated["train"].isdisjoint(updated["test"])
        assert updated["train"].isdisjoint(updated["valid"])
        assert updated["test"].isdisjoint(updated["valid"])
        assert 75 <= len(updated["train"]) <= 85
        assert 5 <= len(updated["test"]) <= 15
        assert 5 <= len(updated["valid"]) <= 15

    def test_verify_no_unseen_peptides_raises(self) -> None:
        """split-only guard fails when a peptide is missing from the registry."""
        split_lookup = {
            "train": {"KNOWN"},
            "test": set(),
            "valid": set(),
        }
        with pytest.raises(ValueError, match="not in registry"):
            verify_no_unseen_peptides(
                parquet_files=[],
                split_lookup=split_lookup,
                dataset_peptides={"KNOWN", "MISSING"},
            )

    def test_verify_no_unseen_peptides_passes(self) -> None:
        """All dataset peptides present in registry → no error."""
        split_lookup = {
            "train": {"A", "B"},
            "test": {"C"},
            "valid": {"D"},
        }
        verify_no_unseen_peptides(
            parquet_files=[],
            split_lookup=split_lookup,
            dataset_peptides={"A", "C"},
        )

    def test_process_directories_preserves_seed_and_writes_shards(self) -> None:
        """E2E: seeded local registry + both mode writes shards without moving seeds."""
        seed = {
            "train": {"SEEDAAA", "SEEDBBB"},
            "test": {"SEEDCCC"},
            "valid": {"SEEDDDD"},
        }
        self._write_registry(self.registry_dir, seed, use_hf_valid=True)

        sequences = [
            "SEEDAAA",
            "SEEDBBB",
            "SEEDCCC",
            "SEEDDDD",
            "NEWPEPAA",
            "NEWPEPBB",
            "NEWPEPCC",
            "NEWPEPDD",
            "NEWPEPEE",
            "NEWPEPFF",
        ]
        self._minimal_labelled_parquet(
            self.data_dir / "batch.parquet",
            sequences=sequences,
            unmodified=sequences,
        )

        process_directories(
            input_dirs=[str(self.data_dir)],
            output_dir=str(self.output_dir),
            rows_per_file=100,
            registry_dir=str(self.registry_dir),
            mode=Mode.BOTH,
            upload_to_hf=False,
        )

        _, updated = load_peptide_registry(str(self.output_dir))
        assert "SEEDAAA" in updated["train"]
        assert "SEEDBBB" in updated["train"]
        assert "SEEDCCC" in updated["test"]
        assert "SEEDDDD" in updated["valid"]

        train_files = list(self.output_dir.glob("train_*.parquet"))
        assert train_files, "expected train shards"
        assert list(self.output_dir.glob("test_*.parquet"))
        assert list(self.output_dir.glob("valid_*.parquet"))

        all_written: set[str] = set()
        for pattern in ("train_*.parquet", "test_*.parquet", "valid_*.parquet"):
            for f in self.output_dir.glob(pattern):
                all_written.update(
                    pl.read_parquet(f)["unmodified_peptide"].to_list()
                )
        assert "SEEDAAA" in all_written
        assert "SEEDCCC" in all_written
        assert "SEEDDDD" in all_written
        assert {"NEWPEPAA", "NEWPEPBB", "NEWPEPCC", "NEWPEPDD", "NEWPEPEE", "NEWPEPFF"} <= all_written


MODULE = Path(__file__).resolve().parents[3] / "scripts" / "splitting" / "split_labelled_data.py"


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text())


def _called_names(tree: ast.Module) -> set[str]:
    """Every function name called anywhere in the module."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def test_snapshot_download_is_not_imported() -> None:
    """The whole-repo downloader should not even be in scope."""
    imported = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "snapshot_download" not in imported, (
        "snapshot_download downloads the entire dataset repo. Use hf_hub_download with "
        "filename=REGISTRY_FILENAME to fetch only the registry."
    )


def test_snapshot_download_is_not_called() -> None:
    """Belt and braces: not reachable via an aliased or attribute call either."""
    assert "snapshot_download" not in _called_names(_tree())


def test_registry_is_fetched_with_hf_hub_download() -> None:
    """The single-file downloader is used, and told which single file."""
    tree = _tree()
    assert "hf_hub_download" in _called_names(tree)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hf_hub_download"
        ):
            kwargs = {kw.arg for kw in node.keywords}
            assert (
                "filename" in kwargs
            ), "hf_hub_download without filename= does not restrict what is fetched"
            filename = next(kw.value for kw in node.keywords if kw.arg == "filename")
            assert (
                isinstance(filename, ast.Name) and filename.id == "REGISTRY_FILENAME"
            ), "pass filename=REGISTRY_FILENAME so the constant stays the single source"
            return
    raise AssertionError("no direct hf_hub_download(...) call found")
