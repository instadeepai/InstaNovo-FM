# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Demo: embed a spectrum with the published encoder, then sequence it de novo.

Runs on the committed sample spectrum by default:

    python scripts/demo.py sample_data/spectra.mgf

Any MGF, mzML, mzXML, CSV or parquet input that ``SpectrumDataFrame`` accepts works
the same way. The two checkpoints are downloaded on first use and cached under
``~/.cache/instanovo-fm/``. See the Demo section of the README for the expected
output and how long it takes.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from instanovo.inference import BeamSearchDecoder
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.model.encoder import FoundationModel
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame
from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo

path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/spectra.mgf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sdf = SpectrumDataFrame.load(path, lazy=False, is_annotated=True)
raw = sdf.to_dataset(in_memory=True)
print(f"Loaded {len(raw)} spectra from {path}")

model, cfg = FoundationModel.from_pretrained("instanovo-fm-v0.1.0")
model = model.to(device).eval()

proc = FoundationalDataProcessor(
    n_peaks=cfg.get("n_peaks", 200),
    min_mz=cfg.get("min_mz", 50.0),
    max_mz=cfg.get("max_mz", 2500.0),
    min_intensity=cfg.get("min_intensity", 1e-6),
    remove_precursor_tol=0.0,
    use_spectrum_utils=cfg.get("use_spectrum_utils", False),
    normalize_mz=cfg.get("normalize_mz", True),
    peak_ordering=cfg.get("peak_ordering", "sorted"),
    annotated=False,
    masking_strategy="none",
)
ds = proc.process_dataset(raw)
batch = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=proc.collate_fn)))
emb = model.encode_mean_pooled(batch["spectra"].to(device))
print(f"Embedding shape: {tuple(emb.shape)}")
print(f"Embedding norm:  {emb[0].norm():.3f}")
first5 = ", ".join(f"{v:.4f}" for v in emb[0].detach().cpu().numpy()[:5])
print(f"First 5 values:  [{first5}]")

dn_model, dn_cfg = DownstreamDeNovo.from_pretrained("instanovo-fm-denovo-v0.1.0")
dn_model = dn_model.to(device).eval()
dn_proc = DownstreamDeNovoDataProcessor(
    residue_set=dn_model.residue_set,
    n_peaks=dn_cfg.get("n_peaks", 200),
    min_mz=dn_cfg.get("min_mz", 50.0),
    max_mz=dn_cfg.get("max_mz", 2500.0),
    min_intensity=dn_cfg.get("min_intensity", 1e-6),
    remove_precursor_tol=dn_cfg.get("remove_precursor_tol", 2.0),
    use_spectrum_utils=dn_cfg.get("use_spectrum_utils", False),
    normalize_mz=dn_cfg.get("normalize_mz", True),
    annotated=False,
    return_str=True,
)
dn_ds = dn_proc.process_dataset(raw)
dn_batch = next(iter(DataLoader(dn_ds, batch_size=len(dn_ds), collate_fn=dn_proc.collate_fn)))
out = BeamSearchDecoder(model=dn_model).decode(
    spectra=dn_batch["spectra"].to(device),
    precursors=dn_batch["precursors"].to(device),
    beam_size=5,
    max_length=dn_cfg.get("max_length", 40),
)
peptide = "".join(out["predictions"][0])
logp = float(np.asarray(out["prediction_log_probability"])[0])
print(f"Predicted peptide: {peptide}")
print(f"Log probability:   {logp:.3f}  (confidence {np.exp(logp):.3f})")
