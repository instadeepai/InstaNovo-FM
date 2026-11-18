# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Demo: what the foundation model does, on the spectra committed in sample_data/.

    python scripts/demo.py

Three steps, each one of the paper's claims at a scale that runs on a laptop:

1. embed one spectrum with the published encoder, and read the fingerprint out;
2. sequence that spectrum de novo with the downstream decoder;
3. retrieve peptide identities for 20 query spectra against a 270-spectrum
   library, by embedding nearest neighbour alone -- database-free identification.

Checkpoints download on first use and cache under ``~/.cache/instanovo-fm/``.
See the Demo section of the README for the expected output and how long it takes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from instanovo.inference import BeamSearchDecoder
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.model.encoder import FoundationModel
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame

SAMPLE_DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_data")
SPECTRUM = SAMPLE_DATA / "fig4_TGVELGKPTHFTVNAK.mgf"
LIBRARY = SAMPLE_DATA / "retrieval_library.mgf"
QUERIES = SAMPLE_DATA / "ninespecies_queries.mgf"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    sdf = SpectrumDataFrame.load(str(path), lazy=False, is_annotated=True)
    return sdf, sdf.to_dataset(in_memory=True)


def embed(dataset, processor, model, batch_size=64):
    out = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=processor.collate_fn):
        out.append(model.encode_mean_pooled(batch["spectra"].to(device)))
    return torch.cat(out)


print(f"Device: {device}")

model, cfg = FoundationModel.from_pretrained("instanovo-fm-v0.1.0")
model = model.to(device).eval()
processor = FoundationalDataProcessor(
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

# 1 -- embed one spectrum.
sdf, raw = load(SPECTRUM)
truth = str(sdf.to_pandas()["sequence"].iloc[0])
emb = embed(processor.process_dataset(raw), processor, model)
first5 = ", ".join(f"{v:.4f}" for v in emb[0].detach().cpu().numpy()[:5])
print(f"\n[1] Embedding {SPECTRUM.name} ({truth})")
print(f"    Shape:          {tuple(emb.shape)}")
print(f"    Norm:           {emb[0].norm():.3f}")
print(f"    First 5 values: [{first5}]")

# 2 -- sequence it de novo.
dn_model, dn_cfg = DownstreamDeNovo.from_pretrained("instanovo-fm-denovo-v0.1.0")
dn_model = dn_model.to(device).eval()
dn_processor = DownstreamDeNovoDataProcessor(
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
dn_ds = dn_processor.process_dataset(raw)
dn_batch = next(iter(DataLoader(dn_ds, batch_size=len(dn_ds), collate_fn=dn_processor.collate_fn)))
decoded = BeamSearchDecoder(model=dn_model).decode(
    spectra=dn_batch["spectra"].to(device),
    precursors=dn_batch["precursors"].to(device),
    beam_size=5,
    max_length=dn_cfg.get("max_length", 40),
)
peptide = "".join(decoded["predictions"][0])
logp = float(np.asarray(decoded["prediction_log_probability"])[0])
print(f"\n[2] De novo sequencing {SPECTRUM.name}")
print(f"    Annotated:  {truth}")
print(f"    Predicted:  {peptide}   ({'match' if peptide == truth else 'MISMATCH'})")
print(f"    Log prob:   {logp:.3f}  (uncalibrated score {np.exp(logp):.3f})")

# 3 -- database-free retrieval: nearest neighbour in embedding space.
lib_sdf, lib_raw = load(LIBRARY)
qry_sdf, qry_raw = load(QUERIES)
lib_emb = embed(processor.process_dataset(lib_raw), processor, model)
qry_emb = embed(processor.process_dataset(qry_raw), processor, model)
lib_seq = np.array([str(s) for s in lib_sdf.to_pandas()["sequence"]])
qry_seq = np.array([str(s) for s in qry_sdf.to_pandas()["sequence"]])

similarity = qry_emb @ lib_emb.T
scores, positions = similarity.max(dim=1)
retrieved = lib_seq[positions.cpu().numpy()]
correct = retrieved == qry_seq
print(f"\n[3] Database-free retrieval: {len(qry_seq)} queries vs {len(lib_seq)}-spectrum library")
print(f"    Top-1 correct:       {correct.sum()}/{len(correct)} ({100 * correct.mean():.0f}%)")
print(f"    Mean cosine (hit):   {scores[torch.as_tensor(correct)].mean():.3f}")
if (~correct).any():
    print(f"    Mean cosine (miss):  {scores[torch.as_tensor(~correct)].mean():.3f}")
for i in range(min(3, len(qry_seq))):
    mark = "OK  " if correct[i] else "MISS"
    print(f"    {mark} query {qry_seq[i]:<20s} -> {retrieved[i]:<20s} cos {scores[i]:.3f}")
