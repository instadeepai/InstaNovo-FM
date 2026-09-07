"""Unit tests for the upstream XuanjiNovo de novo benchmark."""

from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyteomics import mgf

from instanovo_fm.eval import convert_parquet_to_mgf as conv
from instanovo_fm.eval import run_xuanjinovo as runner
from instanovo_fm.eval import score_xuanjinovo as scorer

# Two in-vocabulary targets (kept), one high-charge (dropped), one phospho / out-of-vocab (dropped).
_SYNTHETIC_ROWS = [
    {
        "mz_array": [300.0, 100.0, 200.0],
        "intensity_array": [3.0, 1.0, 2.0],
        "precursor_mz": 500.0,
        "precursor_charge": 2,
        "sequence": "PEPTC[UNIMOD:4]LDEK",
        "frag_type": "HCD",
    },
    {
        "mz_array": [110.0, 220.0],
        "intensity_array": [1.0, 2.0],
        "precursor_mz": 620.0,
        "precursor_charge": 3,
        "sequence": "M[UNIMOD:35]KLSTTQLER",
        "frag_type": "HCD",
    },
    {"mz_array": [111.0, 222.0], "intensity_array": [1.0, 2.0], "precursor_mz": 700.0, "precursor_charge": 15, "sequence": "AAK", "frag_type": "HCD"},
    {
        "mz_array": [130.0, 240.0],
        "intensity_array": [1.0, 2.0],
        "precursor_mz": 810.0,
        "precursor_charge": 2,
        "sequence": "S[UNIMOD:21]PEPK",
        "frag_type": "HCD",
    },
]


class TestTitleRoundTrip:
    """The MGF title encodes the join key, recovered by the scorer's parser."""

    def test_build_title_and_parse_are_inverse(self) -> None:
        """build_title(pid) parses back to the same id as a string (int index, single-dataset mode)."""
        for pid in (0, 5, 137):
            assert scorer.parse_prediction_id(conv.build_title(pid)) == str(pid)

    def test_parse_composite_pid(self) -> None:
        """A composite <dataset>:<index> pid (combined mode) round-trips verbatim, underscores kept."""
        assert scorer.parse_prediction_id(conv.build_title("hela_qc:5")) == "hela_qc:5"
        assert scorer.parse_prediction_id(conv.build_title("gluc:113372")) == "gluc:113372"

    def test_parse_falls_back_to_scans_then_trailing_int(self) -> None:
        """The parser recovers ids from XuanjiNovo _SCANS_ titles and trailing integers (as strings)."""
        assert scorer.parse_prediction_id("C:/data/file.mgf_SCANS_42") == "42"
        assert scorer.parse_prediction_id("something 7") == "7"

    def test_parse_returns_none_when_no_id(self) -> None:
        """A title with no numeric id yields None (dropped with a warning downstream)."""
        assert scorer.parse_prediction_id("no-id-here") is None


class TestConvertParquetToMgf:
    """convert() filters unpredictable rows, writes sorted-peak MGF + a matching targets sidecar."""

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, int]:
        """Convert the synthetic rows to MGF + sidecar, stubbing out the SpectrumDataFrame loader."""
        df = pd.DataFrame(_SYNTHETIC_ROWS)
        monkeypatch.setattr(conv, "load_spectrum_dataframe", lambda *a, **k: df.copy())
        mgf_path = tmp_path / "d.mgf"
        sidecar_path = tmp_path / "d_targets.parquet"
        n = conv.convert("ignored.parquet", str(mgf_path), str(sidecar_path))
        return mgf_path, sidecar_path, n

    def test_filters_charge_and_oov_rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rows with charge > max or out-of-vocab targets are dropped; two predictable rows survive."""
        _, _, n = self._run(tmp_path, monkeypatch)
        assert n == 2

    def test_mgf_titles_and_sorted_peaks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each spectrum has a pid= title, its precursor, and peaks sorted by ascending m/z."""
        mgf_path, _, _ = self._run(tmp_path, monkeypatch)
        spectra = list(mgf.read(str(mgf_path), use_index=False))
        assert [s["params"]["title"] for s in spectra] == ["pid=0", "pid=1"]
        assert int(spectra[0]["params"]["charge"][0]) == 2
        assert spectra[0]["params"]["pepmass"][0] == pytest.approx(500.0)
        assert list(spectra[0]["m/z array"]) == pytest.approx([100.0, 200.0, 300.0])

    def test_sidecar_aligns_with_mgf(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sidecar has one row per kept spectrum, aligned prediction_ids, targets and group."""
        _, sidecar_path, _ = self._run(tmp_path, monkeypatch)
        sidecar = pd.read_parquet(sidecar_path)
        assert list(sidecar["prediction_id"]) == [0, 1]
        assert list(sidecar["targets"]) == ["PEPTC[UNIMOD:4]LDEK", "M[UNIMOD:35]KLSTTQLER"]
        assert list(sidecar["group"]) == ["HCD", "HCD"]


class TestLoadXuanjiNovoPredictions:
    """The denovo.tsv reader parses ids from titles and coerces columns."""

    def _write_tsv(self, tmp_path: Path, rows: list[dict]) -> Path:
        """Write a tab-separated denovo.tsv with the upstream column names."""
        path = tmp_path / "denovo.tsv"
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        return path

    def test_parses_ids_and_scores(self, tmp_path: Path) -> None:
        """Titles become string prediction_ids and scores are numeric."""
        tsv = self._write_tsv(
            tmp_path,
            [
                {"title": "pid=0", "prediction": "AAK", "charge": 2, "score": 0.9},
                {"title": "pid=1", "prediction": "PEPK", "charge": 2, "score": 0.8},
            ],
        )
        df = scorer.load_xuanjinovo_predictions(str(tsv))
        assert list(df["prediction_id"]) == ["0", "1"]
        assert df["score"].tolist() == pytest.approx([0.9, 0.8])

    def test_missing_prediction_column_raises(self, tmp_path: Path) -> None:
        """A tsv missing an expected column raises a clear KeyError."""
        tsv = self._write_tsv(tmp_path, [{"title": "pid=0", "charge": 2, "score": 0.9}])
        with pytest.raises(KeyError):
            scorer.load_xuanjinovo_predictions(str(tsv))


class TestScoreXuanjiNovo:
    """End-to-end: join denovo.tsv to the sidecar, build the canonical CSV, score with the shared harness."""

    def _sidecar(self, tmp_path: Path) -> Path:
        """Write a three-spectrum targets sidecar."""
        path = tmp_path / "sidecar.parquet"
        pd.DataFrame(
            {
                "prediction_id": [0, 1, 2],
                "targets": ["PEPTC[UNIMOD:4]LDEK", "M[UNIMOD:35]KLSTTQLER", "AAK"],
                "precursor_mz": [500.0, 620.0, 300.0],
                "precursor_charge": [2, 3, 2],
                "group": ["all", "all", "all"],
            }
        ).to_parquet(path, index=False)
        return path

    def _tsv(self, tmp_path: Path, rows: list[dict]) -> Path:
        """Write a denovo.tsv with the given rows."""
        path = tmp_path / "denovo.tsv"
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        return path

    def test_scores_raw_notation_with_remapping(self, tmp_path: Path) -> None:
        """Bracketed upstream XuanjiNovo offset predictions score against UNIMOD targets (2/3 correct)."""
        tsv = self._tsv(
            tmp_path,
            [
                {"title": "pid=0", "prediction": "PEPTC[+57.021]LDEK", "charge": 2, "score": 0.9},
                {"title": "pid=1", "prediction": "M[+15.995]KLSTTQLER", "charge": 3, "score": 0.8},
                {"title": "pid=2", "prediction": "AAR", "charge": 2, "score": 0.7},  # wrong
            ],
        )
        results = scorer.score(str(tsv), str(self._sidecar(tmp_path)), str(tmp_path / "pred.csv"), output_dir=str(tmp_path))
        assert results["overall"]["n"] == 3
        assert results["overall"]["peptide_recall"] == pytest.approx(2 / 3)

    def test_missing_prediction_counts_as_miss(self, tmp_path: Path) -> None:
        """A sidecar spectrum with no prediction is predictable=False and a miss under the true denominator."""
        tsv = self._tsv(
            tmp_path,
            [
                {"title": "pid=0", "prediction": "PEPTC[+57.021]LDEK", "charge": 2, "score": 0.9},
                {"title": "pid=1", "prediction": "M[+15.995]KLSTTQLER", "charge": 3, "score": 0.8},
            ],
        )
        results = scorer.score(str(tsv), str(self._sidecar(tmp_path)), str(tmp_path / "pred.csv"))
        assert results["overall"]["n"] == 3
        assert results["overall"]["n_predictable"] == 2
        assert results["overall"]["n_unpredictable"] == 1
        # Both attempted spectra are correct → filtered (predictable-only) recall is 1.0.
        assert results["overall"]["peptide_recall_predictable"] == pytest.approx(1.0)

    def test_canonical_dataframe_log_probs_and_flags(self, tmp_path: Path) -> None:
        """build_canonical_dataframe stores log(score) and flags predictable rows correctly."""
        sidecar = pd.read_parquet(self._sidecar(tmp_path))
        preds = pd.DataFrame({"prediction_id": [0, 2], "prediction": ["AAK", "AAR"], "score": [0.5, 0.25]})
        canonical = scorer.build_canonical_dataframe(preds, sidecar)
        assert list(canonical["predictable"]) == [True, False, True]
        row0 = canonical[canonical["prediction_id"] == "0"].iloc[0]
        assert row0["log_probs"] == pytest.approx(np.log(0.5))
        assert canonical[canonical["prediction_id"] == "1"].iloc[0]["predictions"] == ""


class TestConvertCombined:
    """convert_combined merges datasets into one MGF + sidecar with composite ids and per-dataset groups."""

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, int]:
        """Convert two synthetic datasets into one combined MGF + sidecar, stubbing the loader per path."""
        frames = {
            "alpha.parquet": pd.DataFrame([_SYNTHETIC_ROWS[0], _SYNTHETIC_ROWS[1]]),  # 2 in-vocab rows
            "beta.parquet": pd.DataFrame([_SYNTHETIC_ROWS[0]]),  # 1 in-vocab row
        }
        monkeypatch.setattr(conv, "load_spectrum_dataframe", lambda path, *a, **k: frames[path].copy())
        mgf_path = tmp_path / "combined.mgf"
        sidecar_path = tmp_path / "combined_targets.parquet"
        n = conv.convert_combined([("alpha", "alpha.parquet"), ("beta", "beta.parquet")], str(mgf_path), str(sidecar_path))
        return mgf_path, sidecar_path, n

    def test_total_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The total returned spans all datasets (2 + 1 = 3 surviving rows)."""
        _, _, n = self._run(tmp_path, monkeypatch)
        assert n == 3

    def test_composite_ids_and_groups(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sidecar carries composite <dataset>:<index> ids and the dataset name as group."""
        _, sidecar_path, _ = self._run(tmp_path, monkeypatch)
        sidecar = pd.read_parquet(sidecar_path)
        assert list(sidecar["prediction_id"]) == ["alpha:0", "alpha:1", "beta:0"]
        assert list(sidecar["group"]) == ["alpha", "alpha", "beta"]

    def test_mgf_titles_and_global_scans(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The single MGF carries composite pid titles and globally-unique scan numbers."""
        mgf_path, _, _ = self._run(tmp_path, monkeypatch)
        spectra = list(mgf.read(str(mgf_path), use_index=False))
        assert [s["params"]["title"] for s in spectra] == ["pid=alpha:0", "pid=alpha:1", "pid=beta:0"]
        assert [s["params"]["scans"] for s in spectra] == ["0", "1", "2"]


class TestCombinedScoring:
    """Combined-run scoring: composite-id join, per-group metrics, and InstaNovo-format results CSV."""

    def _sidecar(self, tmp_path: Path) -> Path:
        """Write a combined two-species sidecar keyed by composite ids."""
        path = tmp_path / "combined_sidecar.parquet"
        pd.DataFrame(
            {
                "prediction_id": ["alpha:0", "alpha:1", "beta:0", "beta:1"],
                "targets": ["AAK", "PEPK", "M[UNIMOD:35]KLSTTQLER", "AAR"],
                "precursor_mz": [300.0, 400.0, 620.0, 300.0],
                "precursor_charge": [1, 1, 3, 2],
                "group": ["alpha", "alpha", "beta", "beta"],
            }
        ).to_parquet(path, index=False)
        return path

    def _tsv(self, tmp_path: Path) -> Path:
        """Write a single combined denovo.tsv with composite-pid titles across both species."""
        path = tmp_path / "denovo.tsv"
        pd.DataFrame(
            [
                {"title": "pid=alpha:0", "prediction": "AAK", "charge": 1, "score": 0.9},  # correct
                {"title": "pid=alpha:1", "prediction": "AAR", "charge": 1, "score": 0.8},  # wrong vs PEPK
                {"title": "pid=beta:0", "prediction": "M[+15.995]KLSTTQLER", "charge": 3, "score": 0.7},  # correct (remapped)
                {"title": "pid=beta:1", "prediction": "AAR", "charge": 2, "score": 0.6},  # correct
            ]
        ).to_csv(path, sep="\t", index=False)
        return path

    def test_per_group_and_overall_metrics(self, tmp_path: Path) -> None:
        """Composite ids join correctly and per-group + overall recall are computed."""
        results = scorer.score(
            str(self._tsv(tmp_path)),
            str(self._sidecar(tmp_path)),
            str(tmp_path / "preds.csv"),
            output_dir=str(tmp_path),
        )
        assert set(results["groups"]) == {"alpha", "beta"}
        assert results["groups"]["alpha"]["peptide_recall"] == pytest.approx(0.5)  # 1 of 2
        assert results["groups"]["beta"]["peptide_recall"] == pytest.approx(1.0)  # 2 of 2
        assert results["overall"]["peptide_recall"] == pytest.approx(3 / 4)

    def test_results_csv_wide_format(self, tmp_path: Path) -> None:
        """The results CSV is a single wide row: metadata + {group}_{metric} columns (predictor.py style)."""
        scorer.score(
            str(self._tsv(tmp_path)),
            str(self._sidecar(tmp_path)),
            str(tmp_path / "preds.csv"),
            results_csv=str(tmp_path / "results.csv"),
            run_name="xuanjinovo_test",
            model_label="xuanjinovo_100m",
        )
        row = pd.read_csv(tmp_path / "results.csv")
        assert len(row) == 1
        assert list(row.columns)[:4] == ["run_name", "instanovo_model", "num_beams", "use_knapsack"]
        assert row.loc[0, "run_name"] == "xuanjinovo_test"
        assert row.loc[0, "alpha_pep_recall"] == pytest.approx(0.5)
        assert row.loc[0, "beta_pep_recall"] == pytest.approx(1.0)
        assert {"alpha_pep_prec", "alpha_aa_er", "beta_aa_recall"} <= set(row.columns)


class _FakeS3:
    """Stand-in for :class:`S3FileHandler` that keeps "uploaded" objects in a dict.

    ``land=False`` reproduces the real handler's behaviour when a put fails: ``upload`` swallows the
    exception, so the write appears to succeed while no object is created.
    """

    def __init__(self, objects: dict[str, bytes] | None = None, land: bool = True) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        self.land = land
        self.s3 = self  # the real handler exposes the s3fs instance here; we serve `exists` ourselves

    def exists(self, path: str) -> bool:
        """Report whether a virtual object exists at ``path``."""
        return path in self.objects

    def download(self, s3_path: str, local_path: str) -> None:
        """Write the stored bytes for ``s3_path`` to ``local_path``."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.objects.get(s3_path, b""))

    def upload_to_s3_wrapper(self, save_func, s3_path: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Write locally for local destinations; record a virtual object for s3:// ones."""
        if not s3_path.startswith("s3://"):
            Path(s3_path).parent.mkdir(parents=True, exist_ok=True)
            return save_func(s3_path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as staging:
            staged = Path(staging) / Path(s3_path).name
            result = save_func(str(staged), *args, **kwargs)
            if self.land:
                self.objects[s3_path] = staged.read_bytes()
        return result


class TestRunXuanjiNovoArgs:
    """Argument resolution for the XuanjiNovo runner."""

    def test_positional_arguments_win_over_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positional arguments take precedence over the XUANJINOVO_* environment variables."""
        monkeypatch.setenv("XUANJINOVO_INPUT", "s3://env/in.mgf")
        monkeypatch.setenv("XUANJINOVO_OUTPUT", "s3://env/out")
        args = runner.parse_args(["/local/in.mgf", "/local/out"])
        assert args.input_mgf == "/local/in.mgf"
        assert args.output_prefix == "/local/out"

    def test_environment_supplies_arguments_when_positional_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no positional arguments the environment variables are used."""
        monkeypatch.setenv("XUANJINOVO_INPUT", "s3://env/in.mgf")
        monkeypatch.setenv("XUANJINOVO_OUTPUT", "s3://env/out")
        args = runner.parse_args([])
        assert (args.input_mgf, args.output_prefix) == ("s3://env/in.mgf", "s3://env/out")

    def test_missing_input_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing input MGF aborts rather than running inference on nothing."""
        monkeypatch.delenv("XUANJINOVO_INPUT", raising=False)
        monkeypatch.setenv("XUANJINOVO_OUTPUT", "s3://env/out")
        with pytest.raises(SystemExit):
            runner.parse_args([])

    def test_missing_output_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing output prefix aborts before any GPU time is spent."""
        monkeypatch.setenv("XUANJINOVO_INPUT", "s3://env/in.mgf")
        monkeypatch.delenv("XUANJINOVO_OUTPUT", raising=False)
        with pytest.raises(SystemExit):
            runner.parse_args([])

    def test_tuning_knobs_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The beam/tolerance/batch knobs default from the environment."""
        monkeypatch.setenv("XUANJINOVO_N_BEAMS", "7")
        monkeypatch.setenv("XUANJINOVO_MASS_CONTROL_TOL", "0.25")
        monkeypatch.setenv("XUANJINOVO_BATCH_SIZE", "8")
        args = runner.parse_args(["in.mgf", "out"])
        assert (args.n_beams, args.mass_control_tol, args.batch_size) == (7, 0.25, 8)


class TestRunXuanjiNovoStaging:
    """Destination checks and input staging."""

    def test_s3_output_without_configuration_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An s3:// output prefix with S3 unconfigured fails fast instead of silently writing nothing."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: False))
        with pytest.raises(RuntimeError, match="S3 is not configured"):
            runner.check_destination("s3://bucket/out", allow_local_fallback=False)

    def test_s3_output_without_configuration_warns_when_fallback_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the fallback enabled an unconfigured S3 destination only warns."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: False))
        runner.check_destination("s3://bucket/out", allow_local_fallback=True)

    def test_local_output_prefix_needs_no_s3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A local output prefix is always acceptable."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: False))
        runner.check_destination("/data/out", allow_local_fallback=False)

    def test_local_mgf_is_used_in_place(self, tmp_path: Path) -> None:
        """An existing local MGF is read where it is, with no copy into the work directory."""
        mgf = tmp_path / "biological.mgf"
        mgf.write_text("BEGIN IONS\nEND IONS\n")
        work = tmp_path / "work"
        assert runner.resolve_input_mgf(str(mgf), work, _FakeS3()) == mgf
        assert not work.exists()

    def test_missing_local_mgf_raises(self, tmp_path: Path) -> None:
        """A local MGF that does not exist raises rather than reaching inference."""
        with pytest.raises(FileNotFoundError):
            runner.resolve_input_mgf(str(tmp_path / "absent.mgf"), tmp_path / "work", _FakeS3())

    def test_s3_mgf_without_configuration_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An s3:// MGF with S3 unconfigured raises instead of being passed through as a filename."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: False))
        with pytest.raises(RuntimeError, match="S3 is not configured"):
            runner.resolve_input_mgf("s3://bucket/biological.mgf", tmp_path / "work", _FakeS3())

    def test_s3_mgf_downloads_into_work_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An s3:// MGF is downloaded into the work directory under its own basename."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: True))
        fake = _FakeS3({"s3://bucket/data/biological.mgf": b"BEGIN IONS\n"})
        work = tmp_path / "work"
        resolved = runner.resolve_input_mgf("s3://bucket/data/biological.mgf", work, fake)
        assert resolved == work / "biological.mgf"
        assert resolved.read_bytes() == b"BEGIN IONS\n"

    def test_empty_s3_download_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A download that yields no bytes raises instead of feeding an empty MGF to the model."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: True))
        with pytest.raises(RuntimeError, match="produced no data"):
            runner.resolve_input_mgf("s3://bucket/missing.mgf", tmp_path / "work", _FakeS3())


class TestRunXuanjiNovoPublish:
    """Output discovery and publishing."""

    def _run_dir(self, tmp_path: Path) -> Path:
        """Build an output directory shaped like upstream's timestamped run directory."""
        out_dir = tmp_path / "output"
        run_dir = out_dir / "20260814120000"
        run_dir.mkdir(parents=True)
        (run_dir / "denovo.tsv").write_text("title\tprediction\tcharge\tscore\nx\tPEPTIDE\t2\t0.9\n")
        (run_dir / "train.log").write_text("log line\n")
        return out_dir

    def test_finds_tsv_in_timestamped_subdirectory(self, tmp_path: Path) -> None:
        """The predictions TSV is found inside upstream's timestamped subdirectory."""
        out_dir = self._run_dir(tmp_path)
        assert runner.find_denovo_tsv(out_dir).parent.name == "20260814120000"

    def test_missing_tsv_raises_with_directory_listing(self, tmp_path: Path) -> None:
        """When inference produced no TSV the error names what was there instead."""
        out_dir = tmp_path / "output"
        (out_dir / "20260814120000").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="denovo.tsv not found"):
            runner.find_denovo_tsv(out_dir)

    def test_unlanded_upload_raises(self, tmp_path: Path) -> None:
        """A write that the object store silently dropped is caught rather than reported as success."""
        source = tmp_path / "denovo.tsv"
        source.write_text("data\n")
        with pytest.raises(RuntimeError, match="did not land"):
            runner.write_result(source, "s3://bucket/out/denovo.tsv", _FakeS3(land=False))

    def test_publish_writes_tsv_and_logs_to_local_prefix(self, tmp_path: Path) -> None:
        """A local output prefix receives the TSV and the run logs."""
        out_dir = self._run_dir(tmp_path)
        dest = tmp_path / "published"
        tsv = runner.find_denovo_tsv(out_dir)
        published = runner.publish(tsv, out_dir, str(dest), _FakeS3(), allow_local_fallback=False)
        assert published == str(dest / "denovo.tsv")
        assert (dest / "denovo.tsv").exists()
        assert (dest / "train.log").exists()

    def test_publish_raises_when_tsv_does_not_land(self, tmp_path: Path) -> None:
        """Without the fallback, a dropped TSV write fails the run instead of exiting cleanly."""
        out_dir = self._run_dir(tmp_path)
        with pytest.raises(RuntimeError, match="did not land"):
            runner.publish(runner.find_denovo_tsv(out_dir), out_dir, "s3://bucket/out", _FakeS3(land=False), allow_local_fallback=False)

    def test_publish_keeps_local_copy_when_fallback_allowed(self, tmp_path: Path) -> None:
        """With the fallback enabled a dropped TSV write returns the local path instead of raising."""
        out_dir = self._run_dir(tmp_path)
        tsv = runner.find_denovo_tsv(out_dir)
        published = runner.publish(tsv, out_dir, "s3://bucket/out", _FakeS3(land=False), allow_local_fallback=True)
        assert published == str(tsv)


class TestRunXuanjiNovoMain:
    """End-to-end wiring with inference stubbed out."""

    def test_main_stages_runs_and_publishes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() stages inputs, invokes upstream with the configured knobs, and publishes the TSV."""
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        mgf = tmp_path / "biological.mgf"
        mgf.write_text("BEGIN IONS\nEND IONS\n")
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"weights")
        work = tmp_path / "work"
        dest = tmp_path / "out"
        recorded: dict[str, list[str]] = {}

        def fake_run(command: list[str], cwd: str | None = None, check: bool = False) -> None:
            recorded["command"] = command
            out_dir = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--output=")))
            run_dir = out_dir / "20260814120000"
            run_dir.mkdir(parents=True)
            (run_dir / "denovo.tsv").write_text("title\tprediction\tcharge\tscore\n")
            (run_dir / "train.log").write_text("log\n")

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        exit_code = runner.main(
            [
                str(mgf),
                str(dest),
                "--work_dir",
                str(work),
                "--checkpoint_path",
                str(ckpt),
                "--n_beams",
                "5",
                "--app_dir",
                str(tmp_path),
                "--data_dir",
                str(tmp_path / "data"),
                "--checkpoint_dir",
                str(tmp_path / "ckpt"),
            ]
        )

        assert exit_code == 0
        assert (dest / "denovo.tsv").exists()
        assert f"--peak_path={mgf}" in recorded["command"]
        assert f"--model={ckpt}" in recorded["command"]
        assert "--n_beams=5" in recorded["command"]
        assert "--pmc-enable" in recorded["command"]

    def test_main_wipes_a_previous_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale work directory is cleared, so a previous run's predictions can never be published."""
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        mgf = tmp_path / "biological.mgf"
        mgf.write_text("BEGIN IONS\n")
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"weights")
        work = tmp_path / "work"
        stale = work / "output" / "20200101000000"
        stale.mkdir(parents=True)
        (stale / "denovo.tsv").write_text("STALE\n")

        def fake_run(command: list[str], cwd: str | None = None, check: bool = False) -> None:
            out_dir = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--output=")))
            run_dir = out_dir / "20260814120000"
            run_dir.mkdir(parents=True)
            (run_dir / "denovo.tsv").write_text("FRESH\n")

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        runner.main(
            [
                str(mgf),
                str(tmp_path / "out"),
                "--work_dir",
                str(work),
                "--checkpoint_path",
                str(ckpt),
                "--app_dir",
                str(tmp_path),
                "--data_dir",
                str(tmp_path / "data"),
                "--checkpoint_dir",
                str(tmp_path / "ckpt"),
            ]
        )
        assert (tmp_path / "out" / "denovo.tsv").read_text() == "FRESH\n"


class TestXuanjiNovoImageClosure:
    """The XuanjiNovo image copies a subset of instanovo; the runner must not import beyond it.

    Upstream pins torch 2.1 and InstaNovo requires >2.5, so the package cannot be installed there.
    A new import in any copied module would break the image at runtime with nothing else catching it.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _ENTRYPOINT = "instanovo_fm.eval.run_xuanjinovo"

    def _module_file(self, module: str) -> Path | None:
        """Resolve a dotted instanovo module name to its file, if it exists."""
        rel = module.replace(".", "/")
        for candidate in (self._REPO_ROOT / f"{rel}.py", self._REPO_ROOT / rel / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    def _closure(self) -> set[str]:
        """Walk the runner's transitive import closure within our own packages.

        Parent packages count: importing ``instanovo_fm.eval.run_xuanjinovo`` also executes
        ``instanovo/foundational/__init__.py`` and ``instanovo/foundational/eval/__init__.py``, so a
        heavy import added to one of those breaks the image just as surely as one in the runner.
        """
        seen: set[str] = set()
        queue = [self._ENTRYPOINT]
        while queue:
            module = queue.pop().replace(".__init__", "")
            if module in seen:
                continue
            seen.add(module)
            parts = module.split(".")
            queue.extend(".".join(parts[:i]) for i in range(1, len(parts)))
            path = self._module_file(module)
            if path is None:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                    for alias in node.names:
                        # `from pkg import mod` reaches pkg/mod.py; `from pkg import CONST` does not.
                        submodule = f"{node.module}.{alias.name}"
                        if self._module_file(submodule) is not None:
                            names.append(submodule)
                queue.extend(name for name in names if name.startswith("instanovo"))
        return seen

    def test_runner_closure_is_copied_into_the_image(self) -> None:
        """Every instanovo module the runner reaches is COPY'd into Dockerfile.xuanjinovo."""
        dockerfile = (self._REPO_ROOT / "docker" / "Dockerfile.xuanjinovo").read_text()
        copied = set(re.findall(r"(src/instanovo_fm/[\w/]*\.py)", dockerfile))
        missing = set()
        for module in self._closure():
            path = self._module_file(module)
            if path is None:
                continue
            relative = str(path.relative_to(self._REPO_ROOT))
            if relative not in copied:
                missing.add(relative)
        assert not missing, f"Modules reachable from the runner but not copied into the image: {sorted(missing)}"

    def test_runner_closure_excludes_heavy_dependencies(self) -> None:
        """No module in the closure imports torch, pandas, numpy or model code the image lacks."""
        forbidden = {"torch", "pandas", "numpy", "polars", "datasets", "pytorch_lightning", "faiss"}
        offenders: dict[str, set[str]] = {}
        for module in self._closure():
            path = self._module_file(module)
            if path is None:
                continue
            tree = ast.parse(path.read_text())
            found = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module.split(".")[0])
            if found & forbidden:
                offenders[module] = found & forbidden
        assert not offenders, f"Copied modules import dependencies absent from the XuanjiNovo image: {offenders}"


class TestRunXuanjiNovoDirectories:
    """Directory layout: what is downloaded where, and what gets wiped."""

    def test_download_dirs_default_to_relative_data_and_checkpoints(self) -> None:
        """The download directories default to `data` and `checkpoints`, relative to the working directory."""
        args = runner.parse_args(["in.mgf", "out"])
        assert args.data_dir == Path("data")
        assert args.checkpoint_dir == Path("checkpoints")

    def test_download_dirs_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both download directories can be overridden from the environment."""
        monkeypatch.setenv("XUANJINOVO_DATA_DIR", "/mnt/inputs")
        monkeypatch.setenv("XUANJINOVO_CKPT_DIR", "/mnt/ckpts")
        args = runner.parse_args(["in.mgf", "out"])
        assert (args.data_dir, args.checkpoint_dir) == (Path("/mnt/inputs"), Path("/mnt/ckpts"))

    def test_mgf_download_overwrites_a_stale_copy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A previously downloaded MGF is replaced, so a regenerated dataset is never silently reused."""
        monkeypatch.setattr(runner.S3FileHandler, "s3_enabled", staticmethod(lambda: True))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "biological.mgf").write_bytes(b"STALE")
        fake = _FakeS3({"s3://bucket/biological.mgf": b"FRESH"})
        resolved = runner.resolve_input_mgf("s3://bucket/biological.mgf", data_dir, fake)
        assert resolved.read_bytes() == b"FRESH"

    def test_two_tsvs_raise_rather_than_publishing_the_older(self, tmp_path: Path) -> None:
        """Two run directories are an error, not a choice between a fresh and a stale result."""
        out_dir = tmp_path / "output"
        for stamp in ("20200101000000", "20260814120000"):
            (out_dir / stamp).mkdir(parents=True)
            (out_dir / stamp / "denovo.tsv").write_text(stamp)
        with pytest.raises(RuntimeError, match="Expected exactly one"):
            runner.find_denovo_tsv(out_dir)

    def test_main_wipes_only_the_output_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The per-run wipe clears upstream's output but leaves downloaded inputs and checkpoints intact."""
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        data_dir, ckpt_dir, work = tmp_path / "data", tmp_path / "ckpt", tmp_path / "work"
        for d in (data_dir, ckpt_dir):
            d.mkdir()
        (data_dir / "keep.mgf").write_text("kept")
        (ckpt_dir / "keep.ckpt").write_text("kept")
        stale = work / "output" / "20200101000000"
        stale.mkdir(parents=True)
        (stale / "denovo.tsv").write_text("STALE\n")

        mgf = tmp_path / "biological.mgf"
        mgf.write_text("BEGIN IONS\n")
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"weights")

        def fake_run(command: list[str], cwd: str | None = None, check: bool = False) -> None:
            out = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--output=")))
            (out / "20260814120000").mkdir(parents=True)
            (out / "20260814120000" / "denovo.tsv").write_text("FRESH\n")

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        runner.main(
            [
                str(mgf),
                str(tmp_path / "published"),
                "--work_dir",
                str(work),
                "--checkpoint_path",
                str(ckpt),
                "--app_dir",
                str(tmp_path),
                "--data_dir",
                str(data_dir),
                "--checkpoint_dir",
                str(ckpt_dir),
            ]
        )

        assert (tmp_path / "published" / "denovo.tsv").read_text() == "FRESH\n"
        assert (data_dir / "keep.mgf").exists(), "download directories must not be wiped"
        assert (ckpt_dir / "keep.ckpt").exists(), "download directories must not be wiped"
        assert not (work / "output" / "20200101000000").exists(), "stale run directory must be wiped"

    def test_checkpoint_filename_follows_the_source_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A downloaded checkpoint is named after its URL, so switching URLs cannot reuse the previous file."""
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        ckpt_dir = tmp_path / "ckpt"
        requested: dict[str, Path] = {}

        def fake_download(url: str, dest: Path) -> None:
            requested["dest"] = dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"weights")

        monkeypatch.setattr("instanovo_fm.utils.checkpoints.download_checkpoint", fake_download)

        def fake_run(command: list[str], cwd: str | None = None, check: bool = False) -> None:
            out = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--output=")))
            (out / "20260814120000").mkdir(parents=True)
            (out / "20260814120000" / "denovo.tsv").write_text("x\n")

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        mgf = tmp_path / "biological.mgf"
        mgf.write_text("BEGIN IONS\n")
        runner.main(
            [
                str(mgf),
                str(tmp_path / "published"),
                "--work_dir",
                str(tmp_path / "work"),
                "--app_dir",
                str(tmp_path),
                "--data_dir",
                str(tmp_path / "data"),
                "--checkpoint_dir",
                str(ckpt_dir),
                "--checkpoint_url",
                "https://example.com/models/XuanjiNovo_500M.ckpt",
            ]
        )
        assert requested["dest"] == ckpt_dir / "XuanjiNovo_500M.ckpt"
