"""Unit tests for the Casanovo de novo predictor's preprocessing, remapping and CSV assembly."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from instanovo_fm.eval._predict_de_novo_common import (
    CASANOVO_TO_UNIMOD,
    MODEL_SCORING,
    PREDICTION_CSV_COLUMNS,
    _build_residue_set,
    build_model_vocab,
    build_prediction_dataframe,
    filter_unpredictable_rows,
    reconcile_max_charge,
    score_predictions,
    targets_in_model_vocab,
)
from instanovo_fm.eval.predict_casanovo_de_novo import (
    _casanovo_preprocessing,
    _preprocess_spectrum,
    _resolve_group,
)
from instanovo.utils.residues import ResidueSet

requires_casanovo = pytest.mark.skipif(importlib.util.find_spec("casanovo") is None, reason="the upstream `casanovo` package is not installed")


class TestCasanovoRemappingViaResidueSet:
    """Casanovo's ProForma notation is scored via ResidueSet.residue_remapping (no pre-pass rewrite)."""

    def _residue_set(self) -> ResidueSet:
        """Build the Casanovo scoring residue set (remapping + compound-mod mass)."""
        return _build_residue_set(None, **MODEL_SCORING["casanovo"])

    def test_tokenizes_proforma_mods(self) -> None:
        """Residue-attached and N-terminal ProForma mods tokenise as whole tokens (dash dropped)."""
        rs = self._residue_set()
        assert rs.tokenize("PEPTC[Carbamidomethyl]LDEK") == ["P", "E", "P", "T", "C[Carbamidomethyl]", "L", "D", "E", "K"]
        assert rs.tokenize("[Acetyl]-PEPK") == ["[Acetyl]", "P", "E", "P", "K"]

    def test_remapped_masses_match_unimod(self) -> None:
        """Each Casanovo mod token's mass (via remapping) equals its UNIMOD residue's mass."""
        rs = self._residue_set()
        assert rs.get_mass("C[Carbamidomethyl]") == pytest.approx(rs.get_mass("C[UNIMOD:4]"))
        assert rs.get_mass("M[Oxidation]") == pytest.approx(rs.get_mass("M[UNIMOD:35]"))
        assert rs.get_mass("[Acetyl]") == pytest.approx(rs.get_mass("[UNIMOD:1]"))

    def test_compound_nterm_token_canonicalizes_to_two_mods(self) -> None:
        """The single compound carbamyl+ammonia-loss token expands to two scorable UNIMOD residues."""
        from instanovo_fm.eval._predict_de_novo_common import _canonicalize_peptide

        rs = self._residue_set()
        assert _canonicalize_peptide("[+25.980265]PEPK", rs) == "[UNIMOD:5][UNIMOD:385]PEPK"
        # the canonical form re-tokenises into individually-scorable residues (both in the masses table)
        assert rs.tokenize("[UNIMOD:5][UNIMOD:385]PEPK")[:2] == ["[UNIMOD:5]", "[UNIMOD:385]"]

    def test_scores_raw_predictions_against_unimod_targets(self, tmp_path: Path) -> None:
        """score_predictions with the Casanovo remapping scores raw ProForma predictions correctly."""
        import pandas as pd

        targets = ["PEPTC[UNIMOD:4]LDEK", "M[UNIMOD:35]AK"]
        preds = ["PEPTC[Carbamidomethyl]LDEK", "M[Oxidation]AK"]  # raw Casanovo notation, both correct
        csv = tmp_path / "p.csv"
        pd.DataFrame(
            {
                "prediction_id": range(len(targets)),
                "predictions": preds,
                "targets": targets,
                "log_probs": [-0.1] * len(targets),
                "predictable": [True] * len(targets),
                "precursor_mz": [500.0] * len(targets),
                "precursor_charge": [2] * len(targets),
                "group": ["all"] * len(targets),
            }
        ).to_csv(csv, index=False)
        results = score_predictions(str(csv), **MODEL_SCORING["casanovo"])
        assert results["overall"]["peptide_recall"] == pytest.approx(1.0)


def _casanovo_vocab() -> set[str]:
    """Build Casanovo's canonical residue vocabulary from its residue notation."""
    residues = list("GASPVTCLNDQKEMHFRYW") + [
        "C[Carbamidomethyl]",
        "M[Oxidation]",
        "N[Deamidated]",
        "Q[Deamidated]",
        "[Acetyl]-",
        "[Carbamyl]-",
        "[Ammonia-loss]-",
        "[+25.980265]-",
    ]
    return build_model_vocab(residues, CASANOVO_TO_UNIMOD)


class TestModelVocab:
    """The model vocabulary is built from residue notation, and out-of-vocab targets are flagged."""

    def test_build_model_vocab_canonicalizes_and_collapses_il(self) -> None:
        """Model residues are canonicalised to UNIMOD, the compound mod splits, and I collapses to L."""
        vocab = _casanovo_vocab()
        assert "C[UNIMOD:4]" in vocab and "M[UNIMOD:35]" in vocab
        assert "[UNIMOD:5]" in vocab and "[UNIMOD:385]" in vocab  # compound token contributes both parts
        assert "L" in vocab and "I" not in vocab  # isoleucine collapsed to leucine
        assert "S[UNIMOD:21]" not in vocab  # phospho is not producible by Casanovo

    def test_targets_in_model_vocab(self) -> None:
        """Per-target mask flags representable targets True; empties are True; offenders are tallied."""
        vocab = _casanovo_vocab()
        targets = ["PEPTIDEK", "S[UNIMOD:21]AMPLER", "M[UNIMOD:35]AK", "", "R[UNIMOD:7]K"]
        mask, counts = targets_in_model_vocab(targets, vocab)
        # I->L row, supported-mod row, and the empty (de novo) row are in vocab; phospho + deam-R are not.
        assert mask == [True, False, True, True, False]
        assert counts["S[UNIMOD:21]"] == 1
        assert counts["R[UNIMOD:7]"] == 1


class TestReconcileMaxCharge:
    """The config max_charge is reconciled against the model's (the smaller wins)."""

    def test_none_config_uses_model_value(self) -> None:
        """An unset config max_charge falls back to the model's own."""
        assert reconcile_max_charge(None, 4, "Casanovo") == 4

    def test_config_below_model_restricts(self) -> None:
        """A config max_charge below the model's is honoured (further restriction)."""
        assert reconcile_max_charge(2, 4, "Casanovo") == 2

    def test_config_above_model_clamps_to_model(self) -> None:
        """A config max_charge above the model's is clamped down to the model's limit."""
        assert reconcile_max_charge(8, 4, "Casanovo") == 4


class TestFilterUnpredictableRows:
    """The upfront filter drops charge- and residue-invalid rows entirely (mirrors the predictor)."""

    def _df(self) -> Any:
        """A 5-row frame: valid, charge>max, charge<=0, target-OOV (phospho), and valid again."""
        import pandas as pd

        return pd.DataFrame(
            {
                "precursor_charge": [2, 9, 0, 2, 3],
                "precursor_mz": [500.0] * 5,
                "sequence": ["PEPTIDEK", "AAK", "AAK", "S[UNIMOD:21]AMPLER", "M[UNIMOD:35]AK"],
            }
        )

    def test_drops_invalid_charge_and_oov_targets(self) -> None:
        """Rows with charge outside [1, max_charge] or target residues out of vocab are removed."""
        filtered = filter_unpredictable_rows(self._df(), max_charge=4, model_vocab=_casanovo_vocab(), model_name="Casanovo")
        # kept: row 0 (charge 2, PEPTIDEK), row 4 (charge 3, M[UNIMOD:35]AK); dropped: charge 9, charge 0, phospho.
        assert list(filtered["sequence"]) == ["PEPTIDEK", "M[UNIMOD:35]AK"]

    def test_no_target_column_only_charge_filters(self) -> None:
        """Without a sequence column only the charge filter applies (de novo / unlabelled data)."""
        df = self._df().drop(columns=["sequence"])
        filtered = filter_unpredictable_rows(df, max_charge=4, model_vocab=_casanovo_vocab(), model_name="Casanovo")
        assert list(filtered["precursor_charge"]) == [2, 2, 3]  # only charge 9 and 0 dropped

    def test_config_max_charge_restricts_further(self) -> None:
        """A tighter max_charge drops otherwise-valid higher charges."""
        filtered = filter_unpredictable_rows(self._df(), max_charge=2, model_vocab=_casanovo_vocab(), model_name="Casanovo")
        assert list(filtered["sequence"]) == ["PEPTIDEK"]  # charge-3 row now excluded too


class TestDualDenominatorScoring:
    """Scoring reports both the true denominator (all rows) and the filtered denominator (predictable)."""

    def _write_csv(self, tmp_path: Path) -> str:
        """Write a 4-row CSV: 2 correct predictable, 1 unpredictable (empty pred), 1 predictable miss."""
        import pandas as pd

        csv = tmp_path / "p.csv"
        pd.DataFrame(
            {
                "prediction_id": [0, 1, 2, 3],
                "predictions": ["AK", "PEK", "", "AAK"],
                "targets": ["AK", "PEK", "MLK", "K[UNIMOD:259]R"],
                "log_probs": [-0.1, -0.1, float("nan"), -0.1],
                "predictable": [True, True, False, True],
                "precursor_mz": [500.0] * 4,
                "precursor_charge": [2] * 4,
                "group": ["all"] * 4,
            }
        ).to_csv(csv, index=False)
        return str(csv)

    def test_true_denominator_counts_all_rows(self, tmp_path: Path) -> None:
        """The true denominator is all 4 labelled rows; unpredictable + unscorable rows are misses."""
        overall = score_predictions(self._write_csv(tmp_path), **MODEL_SCORING["casanovo"])["overall"]
        assert overall["n"] == 4  # all labelled rows
        assert overall["n_unpredictable"] == 1  # the empty-prediction (dropped) row
        assert overall["n_target_unscorable"] == 1  # K[UNIMOD:259] target (scored via sentinel as a miss)
        assert overall["peptide_recall"] == pytest.approx(0.5)  # 2 correct of 4

    def test_filtered_denominator_excludes_unpredictable(self, tmp_path: Path) -> None:
        """The filtered denominator is the 3 predictable rows (2 correct, 1 miss)."""
        overall = score_predictions(self._write_csv(tmp_path), **MODEL_SCORING["casanovo"])["overall"]
        assert overall["n_predictable"] == 3
        assert overall["peptide_recall_predictable"] == pytest.approx(2 / 3)


class TestBuildPredictionDataframe:
    """Prediction records are assembled with the canonical schema and deterministic order."""

    def test_schema_matches_reference_prefix(self) -> None:
        """The first five canonical columns match the existing baseline CSV schema."""
        assert PREDICTION_CSV_COLUMNS[:5] == ["prediction_id", "predictions", "targets", "log_probs", "predictable"]

    def test_column_order_and_sort_by_prediction_id(self) -> None:
        """Records are sorted by prediction_id and canonical columns come first."""
        records = [
            {"prediction_id": 2, "predictions": "AAK", "targets": "AAK", "log_probs": -0.1, "predictable": True, "extra": "z"},
            {"prediction_id": 0, "predictions": "PEK", "targets": "PEK", "log_probs": -0.2, "predictable": True, "extra": "x"},
            {"prediction_id": 1, "predictions": "", "targets": "MLK", "log_probs": float("nan"), "predictable": False, "extra": "y"},
        ]
        df = build_prediction_dataframe(records)
        assert list(df["prediction_id"]) == [0, 1, 2]
        assert list(df.columns)[:5] == ["prediction_id", "predictions", "targets", "log_probs", "predictable"]
        assert df.columns[-1] == "extra"  # unknown keys retained, placed last


@requires_casanovo
class TestCasanovoPreprocessing:
    """The native chain is sourced from casanovo's own DeNovoDataModule (exact upstream pipeline)."""

    def test_chain_length_and_valid_charge(self) -> None:
        """_casanovo_preprocessing yields the 6-step chain and a valid-charge set capped at max_charge."""
        chain, valid_charge = _casanovo_preprocessing(max_charge=4, min_intensity=None)
        assert len(chain) == 6
        assert list(valid_charge) == [1, 2, 3, 4]

    def test_full_chain_caps_peaks_and_unit_norms(self) -> None:
        """The sourced chain caps peaks at 150 and yields a unit-norm spectrum."""
        import spectrum_utils.spectrum as sus

        chain, _ = _casanovo_preprocessing(max_charge=4, min_intensity=None)
        mz = np.linspace(100.0, 1500.0, 400)
        intensity = np.random.default_rng(0).uniform(1.0, 10.0, 400).astype(np.float32)
        spectrum = sus.MsmsSpectrum("test", 800.0, 2, mz, intensity)
        for fn in chain:
            spectrum = fn(spectrum)
        assert len(spectrum.mz) <= 150
        assert math.isclose(float(np.linalg.norm(spectrum.intensity)), 1.0, rel_tol=1e-5)


@requires_casanovo
class TestPreprocessSpectrumRow:
    """Row-level preprocessing marks unpredictable spectra as None."""

    def _chain(self) -> tuple:
        """Source the casanovo native chain + valid-charge set for max_charge=4."""
        return _casanovo_preprocessing(max_charge=4, min_intensity=None)

    def _row(self, n_peaks: int, charge: int) -> dict:
        """Build a minimal pandas-row-like dict for _preprocess_spectrum."""
        return {
            "prediction_id": 0,
            "precursor_mz": 800.0,
            "precursor_charge": charge,
            "mz_array": np.linspace(100.0, 1500.0, n_peaks),
            "intensity_array": np.linspace(1.0, 2.0, n_peaks).astype(np.float32),
        }

    def test_rejects_out_of_range_charge(self) -> None:
        """A charge outside valid_charge yields None (unpredictable)."""
        import spectrum_utils.spectrum as sus

        chain, valid_charge = self._chain()
        assert _preprocess_spectrum(self._row(50, charge=9), sus, chain, valid_charge) is None

    def test_rejects_low_quality(self) -> None:
        """A spectrum with too few peaks yields None (unpredictable)."""
        import spectrum_utils.spectrum as sus

        chain, valid_charge = self._chain()
        assert _preprocess_spectrum(self._row(5, charge=2), sus, chain, valid_charge) is None

    def test_accepts_valid_spectrum(self) -> None:
        """A valid in-range spectrum is preprocessed and returned."""
        import spectrum_utils.spectrum as sus

        chain, valid_charge = self._chain()
        spectrum = _preprocess_spectrum(self._row(50, charge=2), sus, chain, valid_charge)
        assert spectrum is not None
        assert len(spectrum.mz) >= 20


class TestScoringHandlesEmptyPredictions:
    """Empty predictions are contained in the de novo scoring layer (fillna) and score as misses."""

    def _residue_set(self) -> ResidueSet:
        """Build a minimal ResidueSet for scoring tests."""
        return ResidueSet(residue_masses={"A": 71.03711, "K": 128.09496, "P": 97.05276, "E": 129.04259})

    def test_tokenize_empty_string_returns_empty(self) -> None:
        """An empty-string prediction (the post-fillna form) tokenises to an empty peptide."""
        assert self._residue_set().tokenize("") == []

    def test_precision_recall_empty_string_is_miss(self) -> None:
        """An empty-string prediction alongside a correct one yields peptide recall 0.5 (a miss)."""
        from instanovo.utils.metrics import Metrics

        metrics = Metrics(residue_set=self._residue_set())
        _aa_prec, _aa_recall, pep_recall, _pep_prec = metrics.compute_precision_recall(["AK", "PEK"], ["AK", ""])
        assert pep_recall == 0.5

    def test_score_predictions_tolerates_blank_prediction_cells(self, tmp_path: Path) -> None:
        """score_predictions fills NaN predictions so a blank cell scores as a miss, not a crash."""
        import pandas as pd

        from instanovo_fm.eval._predict_de_novo_common import score_predictions

        # Mix modified + unmodified peptides so run_all_analyses' mod-presence buckets are both
        # populated (an empty bucket would ZeroDivision inside the analyzer on tiny synthetic data).
        peptides = [
            "PEPTIDEK",
            "ACDEFGHIK",
            "M[UNIMOD:35]SAMPLER",
            "C[UNIMOD:4]VWLYTASK",
            "GGADEK",
            "M[UNIMOD:35]LLNQPK",
            "FHKREK",
            "C[UNIMOD:4]MKTESK",
        ]
        preds = list(peptides)
        preds[1] = ""  # a blank prediction cell -> NaN on read -> must be filled + counted as a miss
        csv_path = tmp_path / "preds.csv"
        pd.DataFrame(
            {
                "prediction_id": range(len(peptides)),
                "predictions": preds,
                "targets": peptides,
                "log_probs": [-0.1] * len(peptides),
                "predictable": [True] * len(peptides),
                "precursor_mz": [500.0] * len(peptides),
                "precursor_charge": [2] * len(peptides),
                "group": ["all"] * len(peptides),
            }
        ).to_csv(csv_path, index=False)

        results = score_predictions(str(csv_path), output_dir=str(tmp_path / "out"))
        assert results["overall"]["n"] == len(peptides)
        assert results["overall"]["peptide_recall"] < 1.0  # the blank prediction is a miss, no crash


class TestConfigDrivenResolution:
    """The config-driven runner resolves model/checkpoint and normalises data_path into groups."""

    def test_resolve_model_and_checkpoint_requires_explicit_model(self) -> None:
        """A config with no ``model`` fails fast (both inference configs set it explicitly)."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _resolve_model_and_checkpoint

        with pytest.raises(ValueError, match="model must be one of"):
            _resolve_model_and_checkpoint(OmegaConf.create({}))

    def test_resolve_model_and_checkpoint_fills_registry_defaults(self) -> None:
        """With an explicit model, checkpoint/url fall back to the registry when omitted."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import MODELS, _resolve_model_and_checkpoint

        model, checkpoint, url = _resolve_model_and_checkpoint(OmegaConf.create({"model": "casanovo"}))
        assert model == "casanovo"
        assert checkpoint == MODELS["casanovo"].default_checkpoint
        assert url == MODELS["casanovo"].checkpoint_url

    def test_resolve_model_and_checkpoint_config_overrides_registry(self) -> None:
        """A config `checkpoint` wins over the registry default."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _resolve_model_and_checkpoint

        _model, checkpoint, _url = _resolve_model_and_checkpoint(OmegaConf.create({"model": "casanovo", "checkpoint": "/tmp/my.ckpt"}))
        assert checkpoint == "/tmp/my.ckpt"

    def test_resolve_model_and_checkpoint_rejects_unknown_model(self) -> None:
        """An unregistered model fails fast."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _resolve_model_and_checkpoint

        with pytest.raises(ValueError, match="model must be one of"):
            _resolve_model_and_checkpoint(OmegaConf.create({"model": "bogus"}))

    def test_normalize_data_groups_string_wraps_single_group(self) -> None:
        """A string data_path becomes one group with result_name/input_path/output_path."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _normalize_data_groups

        cfg = OmegaConf.create({"data_path": "data/foo.parquet", "output_path": "out/foo.csv"})
        groups = _normalize_data_groups(cfg, run_name="myrun")
        assert len(groups) == 1
        assert groups[0]["result_name"] == "myrun"
        assert groups[0]["input_path"] == "data/foo.parquet"
        assert groups[0]["output_path"] == "out/foo.csv"

    def test_normalize_data_groups_list_passthrough(self) -> None:
        """A list data_path (grouped benchmark) is returned as-is."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _normalize_data_groups

        cfg = OmegaConf.create(
            {
                "data_path": [
                    {"result_name": "a", "input_path": "a.parquet", "output_path": "a.csv"},
                    {"result_name": "b", "input_path": "b.parquet", "output_path": "b.csv"},
                ]
            }
        )
        groups = _normalize_data_groups(cfg, run_name="myrun")
        assert [g["result_name"] for g in groups] == ["a", "b"]

    def test_normalize_data_groups_requires_data_path(self) -> None:
        """A missing data_path raises a clear ValueError."""
        from omegaconf import OmegaConf

        from instanovo_fm.eval.run_baseline_de_novo import _normalize_data_groups

        with pytest.raises(ValueError, match="data_path is required"):
            _normalize_data_groups(OmegaConf.create({}), run_name="r")

    def test_build_result_row_flattens_all_per_group_metrics(self) -> None:
        """_build_result_row emits every metric in each dataset's dict as a {group}_{metric} column."""
        from instanovo_fm.eval.run_baseline_de_novo import _build_result_row

        row = _build_result_row(
            "run1",
            "casanovo",
            5,
            {
                "yeast": {"peptide_recall": 0.6, "aa_error_rate": 0.2, "n": 100, "auc": 0.55},
                "human": {"peptide_recall": 0.5, "aa_error_rate": 0.25, "n": 50},
            },
        )
        assert row["run_name"] == "run1"
        assert row["model"] == "casanovo"
        assert row["num_beams"] == 5
        # Keys are the raw metric names (no renaming), and *every* metric flows through.
        assert row["yeast_peptide_recall"] == 0.6
        assert row["yeast_n"] == 100
        assert row["yeast_auc"] == 0.55
        assert row["human_aa_error_rate"] == 0.25


class _StubS3:
    """Minimal S3FileHandler stand-in for the write-fallback tests."""

    def __init__(self, s3: object, raise_on_upload: bool = False) -> None:
        """Store the (possibly None) s3 client and whether uploads should raise."""
        self.s3 = s3
        self._raise = raise_on_upload

    def upload_to_s3_wrapper(self, save_func: Any, dest: str, **kwargs: Any) -> Any:
        """Raise (simulating an S3 failure) or invoke save_func on the local/dest path."""
        if self._raise:
            raise PermissionError("The AWS Access Key Id you provided does not exist in our records.")
        return save_func(dest, **kwargs)


class TestS3WriteFallback:
    """S3 write failures / unconfigured S3 fall back to the already-written local copy (no crash)."""

    def _df_and_local(self, tmp_path: Path) -> tuple:
        """Build a tiny DataFrame and write its local copy; return (df, local_path)."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2]})
        local = tmp_path / "local.csv"
        df.to_csv(local, index=False)
        return df, str(local)

    def test_local_dest_writes_normally(self, tmp_path: Path) -> None:
        """A local destination is written and returned as-is."""
        from instanovo_fm.eval.run_baseline_de_novo import _write_predictions_to_dest

        df, local = self._df_and_local(tmp_path)
        dest = str(tmp_path / "dest.csv")
        out = _write_predictions_to_dest(df, dest, local, _StubS3(s3=object()), allow_fallback=True, label="x")
        assert out == dest
        assert (tmp_path / "dest.csv").exists()

    def test_s3_failure_falls_back_to_local(self, tmp_path: Path) -> None:
        """A failing S3 upload falls back to the local copy instead of raising."""
        from instanovo_fm.eval.run_baseline_de_novo import _write_predictions_to_dest

        df, local = self._df_and_local(tmp_path)
        out = _write_predictions_to_dest(df, "s3://bucket/x.csv", local, _StubS3(s3=object(), raise_on_upload=True), allow_fallback=True, label="x")
        assert out == local

    def test_s3_unconfigured_falls_back_to_local(self, tmp_path: Path) -> None:
        """An s3:// dest with no S3 client configured falls back to the local copy without attempting a write."""
        from instanovo_fm.eval.run_baseline_de_novo import _write_predictions_to_dest

        df, local = self._df_and_local(tmp_path)
        out = _write_predictions_to_dest(df, "s3://bucket/x.csv", local, _StubS3(s3=None), allow_fallback=True, label="x")
        assert out == local

    def test_no_fallback_reraises(self, tmp_path: Path) -> None:
        """With allow_fallback=False a failing S3 upload propagates."""
        from instanovo_fm.eval.run_baseline_de_novo import _write_predictions_to_dest

        df, local = self._df_and_local(tmp_path)
        with pytest.raises(PermissionError):
            _write_predictions_to_dest(df, "s3://bucket/x.csv", local, _StubS3(s3=object(), raise_on_upload=True), allow_fallback=False, label="x")


class TestOovAndResidueLogging:
    """Out-of-vocab targets are flagged with a per-residue tally; residue logging runs cleanly."""

    def test_targets_in_vocab_flags_and_counts_offenders(self) -> None:
        """_targets_in_vocab returns a mask and counts each out-of-vocab residue per target."""
        import pandas as pd

        from instanovo_fm.eval._predict_de_novo_common import _build_residue_set, _targets_in_vocab

        targets = pd.Series(["PEPTIDEK", "C[UNIMOD:312]PEK", "M[UNIMOD:35]AK", "K[UNIMOD:259]R"])
        mask, oov_counts = _targets_in_vocab(targets, _build_residue_set(None))
        assert list(mask) == [True, False, True, False]  # standard + supported mod pass; exotic mods fail
        assert oov_counts["C[UNIMOD:312]"] == 1
        assert oov_counts["K[UNIMOD:259]"] == 1

    def test_log_supported_residues_runs(self) -> None:
        """log_supported_residues splits standard amino acids from modifications without error."""
        from instanovo_fm.eval._predict_de_novo_common import log_supported_residues

        log_supported_residues({"A": 71.03711, "M[UNIMOD:35]": 147.0354, "[UNIMOD:1]": 42.010565}, "TestModel")


class TestResolveGroup:
    """Group labelling prefers frag_type, then experiment_name, then 'all'."""

    def test_prefers_frag_type(self) -> None:
        """frag_type wins when present and non-empty."""
        assert _resolve_group({"frag_type": "HCD", "experiment_name": "expA"}, ["frag_type", "experiment_name"]) == "HCD"

    def test_falls_back_to_experiment_then_all(self) -> None:
        """Falls back to experiment_name, then to 'all' when nothing usable is present."""
        assert _resolve_group({"frag_type": None, "experiment_name": "expA"}, ["frag_type", "experiment_name"]) == "expA"
        assert _resolve_group({"frag_type": None, "experiment_name": None}, ["frag_type", "experiment_name"]) == "all"
