# Downstream de novo sequencing

The downstream task that turns the self-supervised foundation model into a peptide sequencer.
`DownstreamDeNovo` is the **FM encoder plus an InstaNovo autoregressive decoder**: encoder weights
transfer from a foundation checkpoint, the decoder is new, and a staged unfreeze schedule keeps the
pretrained encoder frozen while the decoder learns.

```
foundation checkpoint (encoder only)
        │  load_encoder_from_foundation
        ▼
DownstreamDeNovo  =  peak_encoder + encoder (+ pairwise_bias, ion_ladder, meta_embed)
                     + aa_embed + decoder + head
        │
   train ──> model_best.ckpt ──> predict ──> predictions CSV + metrics
                                    └──────> encoder embeddings (fm_de_novo eval path)
```

| File | Role |
|------|------|
| `model.py` | `DownstreamDeNovo` — architecture, encoder weight transfer, `Decodable` interface |
| `data.py` | `DownstreamDeNovoDataProcessor` — foundation spectrum preprocessing + peptide tokenisation |
| `train.py` | `DownstreamDeNovoTrainer` — subclasses `TransformerTrainer`; wires model + processors |
| `predict.py` | `DownstreamDeNovoPredictor` — subclasses `TransformerPredictor`; loads the checkpoint |
| `cli.py` | `instanovo denovo train` / `instanovo denovo predict` |

Training and inference reuse InstaNovo's `AccelerateDeNovoTrainer` and `TransformerPredictor`
wholesale. The subclasses only override `setup_model`, `setup_data_processors` / `load_model`,
`setup_data_processor` — so the training loop, checkpointing, MLflow tracking, decoders and metrics
are the same code paths the supervised InstaNovo models use, and results are directly comparable.

---

## 1. Training

```bash
uv run accelerate launch --config_file ./src/instanovo_fm/configs/accelerate/aichor.yaml \
  instanovo-fm denovo train
```

Both commands compose from this package's `configs/`, with the installed `instanovo`
package's configs on Hydra's search path so the inherited `instanovo.yaml`,
`model: instanovo_base` and `dataset: default` resolve -- see
[`utils/hydra_config.py`](../../utils/hydra_config.py).

Config: [`configs/denovo.yaml`](../../configs/denovo.yaml), which inherits `instanovo.yaml` (the supervised training
recipe) and overrides the model to `denovo_base`.

### Architecture must match the foundation checkpoint

`configs/model/denovo_base.yaml` inherits `instanovo_base` and then overrides the encoder side to be
byte-identical to the FM:

```yaml
n_head: 12            # 768 / 12 = 64 per head, as in the FM
dim_feedforward: 3072
encoder_layers: 12
normalize_mz: true    # must match the encoder
peak_encoder:
  type: "multiscale"  # must match the FM's peak encoder type and config
```

### Encoder weight transfer

`use_pretrained_model: true` triggers `load_encoder_from_foundation(foundational_checkpoint)`, which
copies only these prefixes out of the foundation state dict:

```
peak_encoder.  encoder.  pairwise_bias.  ion_ladder.  meta_embed.  latent_token  pad_token
```

`mz_head` and every other pretraining head is ignored — the decoder and `head` start random.

The transfer is deliberately loud rather than lenient. Two guards fire before any training step:

- **Structural check** — if the downstream model has an `encoder.*` / `pairwise_bias.*` / `ion_ladder.*`
  submodule but the foundation state has no matching key, it raises. That is the signature of
  `peak_encoder.type`, `ion_ladder.enabled` or `architecture.*` drift between the two configs.
- **Shape check** — a shape mismatch on any transferable key raises rather than skipping. A skipped
  key would silently leave that tensor randomly initialised, which looks like a training problem
  rather than a config problem.

It then logs how many parameters loaded, and any foundation keys with no downstream slot.

Set `use_pretrained_model: false` to train the same architecture from scratch.

### Staged unfreezing

`config.finetune` activates `FinetuneScheduler` (`instanovo/common/scheduler.py`), constructed in
`AccelerateDeNovoTrainer.__init__` and stepped once per training step. It freezes **everything**,
then unfreezes by `fnmatch` pattern at the scheduled step:

```yaml
finetune:
  unfreeze_format: start_step
  unfreeze_schedule:
    - start_step: 0
      params: [decoder.*, head.*, aa_embed.*, aa_pos_embed.*]
    - start_step: 100_000
      params: [peak_encoder.*, encoder.*]
```

Phase 1 trains only the new decoder against a frozen encoder, so early large decoder gradients
cannot wreck the pretrained representation. Phase 2 unfreezes the encoder for joint finetuning.
`unfreeze_format` also accepts `start_epoch`, `duration_steps`, `duration_epochs`; the schedule must
start at step 0 and be monotonically increasing, both validated at construction.

### Meta token

`use_meta_token` / `meta_token.enabled` are **False** in `denovo_base.yaml`. When enabled,
`_add_special_tokens` inserts meta tokens after the latent token, and `setup_data_processors` narrows
`metadata_columns` to just `{frag_type, collision_energy, filepath}` for the train processor —
dataset configs list 30+ columns and building them all per batch is wasted work. The validation
processor keeps the full column list for analysis. `filepath` is there for the `SearchDataManager`
USI lookup, which is only constructed when `dataset.use_search_data` is set.

### Data processing

`DownstreamDeNovoDataProcessor` is the foundation spectrum pipeline plus peptide tokenisation:

1. filter / normalise / scale peaks
2. pad to a fixed `n_peaks` (200)
3. order peaks by `peak_ordering` — `sorted` (default), `cyclic_shift` or `complete_shuffle`
4. tokenise and pad the peptide (reversed for autoregressive decoding, EOS appended)
5. optional metadata enrichment for the meta token

The `peak_ordering` alternatives exist as ablations: `sorted` preserves the spectral pattern,
`cyclic_shift` is augmentation, `complete_shuffle` tests whether the model is using peak order at all.

### Key training parameters

| Key | Value in `denovo.yaml` | Note |
|-----|------------------------|------|
| `learning_rate` | `5e-5` | low — finetuning a pretrained encoder |
| `lr_scheduler` | `cosine_warmup_hold` | linear warmup → hold at max → cosine decay |
| `warmup_iters` / `lr_hold_steps` | `100_000` / `100_000` | hold spans phase 1, so the encoder unfreezes at full LR |
| `train_batch_size` | `128` | |
| `training_steps` | `2_500_000` | |
| `gradient_clip_val` | `10.0` | |
| `model_save_folder_path` | `checkpoints/instanovo-downstream-denovo` | |


Checkpoints, validation cadence (`validation_interval`, `checkpoint_interval`) and best-checkpoint
selection are all inherited from `AccelerateDeNovoTrainer`.

---

## 2. Inference

```bash
uv run accelerate launch --config_file ./src/instanovo_fm/configs/accelerate/aichor.yaml \
  instanovo-fm denovo predict \
  "data_path='<data-root>/lcfm_splits/test_[01].parquet'" \
  denovo_model=s3://…/checkpoints/instanovo-downstream-denovo/model_best.ckpt \
  num_beams=5 use_knapsack=False denovo=False refine=False \
  output_path=s3://…/evaluation/fm_downstream_denovo_lcfm.csv
```

Config: [`configs/inference/denovo.yaml`](../../configs/inference/denovo.yaml), the default for `denovo predict`.

`--denovo` vs `--evaluation` is the mode switch: `denovo=True` writes predictions only,
`denovo=False` (evaluation) also scores against the `sequence` targets and logs metrics to MLflow.
`denovo=True` requires an `output_path`.

`--denovo-model` accepts a local `.ckpt`, an `s3://` URI, or a pretrained model ID; a value
containing `/` or ending `.ckpt` is treated as a path (and must exist), anything else is resolved
against `InstaNovo.get_pretrained()`.

The model is loaded via `DownstreamDeNovo.load()` — **not** `FoundationModel.load()`, which would
crash on a checkpoint that has no `mz_head` and would rebuild the wrong architecture. `load()` reads
the architecture and residue set out of the checkpoint itself, so inference cannot be run against a
mismatched config. On MPS it overrides `peak_embedding_dtype` to float32.

### Decoders

`num_beams` and `use_knapsack` select the decoder in `TransformerPredictor.setup_decoder`:

| Config | Decoder |
|--------|---------|
| `use_knapsack=True` | `KnapsackBeamSearchDecoder` — beams constrained to reachable residue masses (knapsack generated and cached if `knapsack_path` is unset) |
| `num_beams > 1` | `BeamSearchDecoder` |
| `num_beams = 1` | `GreedyDecoder` with basic filtering |

`suppressed_residues` and `disable_terminal_residues_anywhere` (which stops N-terminal mods being
emitted mid-sequence) only apply on the greedy path.

`DownstreamDeNovo` implements the `Decodable` interface — `init`, `score_candidates`,
`get_residue_masses`, `get_eos_index`, `get_empty_index` — which is what lets InstaNovo's existing
decoders drive it unchanged. Note that `_decoder` prepends the precursor (mass encoding + charge
embedding) to the encoder memory, so precursor information reaches the decoder as cross-attention
context even though `_encoder` never sees it in the `fm_de_novo` embedding path.

### Multi-dataset runs

`data_path` in `inference/denovo.yaml` is a **list** of `{result_name, input_path, output_path}`
groups covering the 8 biological validation datasets and the 9 ninespecies-v1 datasets. One job
sequences all of them, writing a per-dataset CSV plus an aggregated row in `result_file_path`.

### Outputs

Per-dataset CSV columns are set by `prediction_col`, `log_probs_col`, `token_log_probs_col`,
`prediction_tokenised_col`, plus whichever of `index_columns` the input carried. `save_beams` keeps
all beams, `save_all_predictions` keeps unfiltered rows. Metrics use `filter_precursor_ppm: 20`,
`filter_confidence: 1e-4` and `filter_fdr_threshold: 0.05`; those filters affect the reported metrics
only, never the saved predictions.

---

## 3. Evaluating the trained encoder

To ask what the *encoder* learned after downstream training — rather than how well it sequences —
run it through the shared baseline embedding path with `MODEL=fm_de_novo`:

```bash
MODEL=fm_de_novo EVAL_DATASET=lcfm TASKS=linearprobetask \
  CHECKPOINT=s3://…/checkpoints/instanovo-downstream-denovo/model_best.ckpt \
  bash scripts/run_baseline_eval.sh
```

That extracts pooled encoder embeddings into the same HDF5 + FAISS layout and runs the same
duplicate-retrieval / linear-probe tasks as every other model, so the finetuned encoder can be
compared against the FM it started from and against the InstaNovo / Casanovo baselines. Details and
caveats: [`../../eval/README_baseline_embeddings.md`](../../eval/README_baseline_embeddings.md).

The encoder token layout there is `[latent(0), peak_1 … peak_N]` — the latent token is **prepended**,
unlike standard InstaNovo which appends it — and m/z is **not** denormalised, because the FM encoder
expects the normalised values `FoundationalDataProcessor` produces.
