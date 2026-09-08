#!/usr/bin/env bash
# =============================================================================
# Baseline UMAP + EVōC clustering figures (Casanovo / InstaNovo)
#
# Two tasks are rendered (both have requires_model=False, so they run on the
# saved embeddings — no live model needed):
#   * umapvisualisationtask  -> umap_{frag_type,search_instrument,search_organism}.png
#                               + umap_summary_panel.png
#   * evocclusteringtask     -> evoc_figs/{evoc_clusters_umap,evoc_enrichment_heatmap,
#                               evoc_cluster_zoom_recolor,evoc_layer_panel,evoc_zoom_cascade}.png
#
# Usage (env-var driven):
#   MODEL=casanovo EVAL_DATASET=lcfm bash scripts/run_baseline_umaps.sh                   # AIchor
#   MODEL=instanovo EVAL_DATASET=mcfm_local DEVICE=cpu bash scripts/run_baseline_umaps.sh # local
#   MODEL=xuanjinovo PREPROCESSING=native EVAL_DATASET=lcfm bash scripts/run_baseline_umaps.sh
#
# Parameters (all overridable via environment):
#   MODEL             casanovo | instanovo | xuanjinovo (required)
#   EVAL_DATASET      dataset-config name               (default: lcfm)
#                     The literal filename of src/instanovo_fm/configs/dataset/<EVAL_DATASET>.yaml
#                     (e.g. lcfm, mcfm, mcfm_local, hcfm) — matches Hydra `dataset=` in the FM
#                     runs. Paths are read from that config (single source of truth); for local
#                     mcfm data use EVAL_DATASET=mcfm_local.
#   SPLIT             valid | test | train                (default: valid)
#                     Which dataset split the UMAP + EVōC figures are rendered on. Reads the
#                     matching <SPLIT>_path from the dataset config; embeddings are extracted
#                     into ${OUTPUT_DIR}/<SPLIT> so runs on different splits don't overwrite.
#   MAX_SAMPLES       embeddings to extract + plot       (default: 100000)
#   POOLING           mean_peaks (default) | native summary token
#                     (casanovo=cls, instanovo=latent, xuanjinovo=precursor)
#   PREPROCESSING     foundation (default) | native — only casanovo/xuanjinovo support 'native'
#                     (each model's own pipeline); ignored for instanovo. Output dir is tagged.
#   DEVICE            cuda | cpu                          (default: cuda)
#   BATCH_SIZE        inference batch size               (default: 256)
#   NUM_WORKERS       dataloader workers                 (default: 4)
#   EVAL_CONFIG_NAME  evaluation config to source         (default: default)
#   MIN_INTENSITY     override preprocessing intensity threshold (default: extractor default 0.01).
#                     When set, it is passed to the extractor and the output dir is tagged
#                     "_mi<value>" so it doesn't overwrite the default run.
#   CHECKPOINT        override checkpoint path            (default: per-model canonical ckpt)
#   SPLIT_PARQUET     override the resolved <SPLIT>-split parquet/glob
#                     (VALID_PARQUET is still honoured when SPLIT=valid, for backward compat)
#   OUTPUT_DIR        override output directory           (default: .../baselines/<dataset>/<model>)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Parameters
# ---------------------------------------------------------------------------- #
MODEL="${MODEL:?MODEL must be set to 'casanovo', 'instanovo' or 'xuanjinovo'}"
EVAL_DATASET="${EVAL_DATASET:-lcfm}"
SPLIT="${SPLIT:-valid}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"

case "$SPLIT" in
  valid|test|train) ;;
  *) echo "ERROR: SPLIT must be 'valid', 'test' or 'train', got '$SPLIT'" >&2; exit 1 ;;
esac
POOLING="${POOLING:-mean_peaks}"
# Spectrum preprocessing: 'foundation' (shared FoundationalDataProcessor) or 'native' (each
# baseline's own pipeline). Only casanovo/xuanjinovo support 'native'; ignored for instanovo.
PREPROCESSING="${PREPROCESSING:-foundation}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-default}"

OUTPUT_BASE="embed_eval_results"

# ---------------------------------------------------------------------------- #
# Resolve dataset -> <SPLIT>-split parquet path FROM THE DATASET CONFIG.
# The src/instanovo_fm/configs/dataset/<EVAL_DATASET>.yaml is the single source of truth,
# so the path can never drift from the FM's Hydra `dataset=` runs.
# ---------------------------------------------------------------------------- #
read -r CFG_SPLIT_PATH CFG_MAX_MZ < <(
  uv run python -c "
from instanovo_fm.eval.run_tasks_from_embeddings import _load_dataset_paths, _load_foundation_max_mz
print(_load_dataset_paths('${EVAL_DATASET}').get('${SPLIT}_path') or '', _load_foundation_max_mz())" | tail -n 1
)
# Backward compat: VALID_PARQUET still overrides the valid split.
if [[ "$SPLIT" == "valid" && -n "${VALID_PARQUET:-}" ]]; then
  SPLIT_PARQUET="$VALID_PARQUET"
fi
SPLIT_PARQUET="${SPLIT_PARQUET:-$CFG_SPLIT_PATH}"
# m/z normalisation divisor — sourced from the foundation-model config (single source of truth).
MAX_MZ="${MAX_MZ:-$CFG_MAX_MZ}"

if [[ -z "$SPLIT_PARQUET" ]]; then
  echo "ERROR: could not resolve a ${SPLIT}_path for EVAL_DATASET='${EVAL_DATASET}'." >&2
  echo "       Expected src/instanovo_fm/configs/dataset/${EVAL_DATASET}.yaml with a '${SPLIT}_path' key." >&2
  exit 1
fi

# ---------------------------------------------------------------------------- #
# Resolve model -> extraction module + canonical checkpoint + download url
# ---------------------------------------------------------------------------- #
case "$MODEL" in
  casanovo)
    EXTRACT_MODULE="instanovo_fm.eval.extract_casanovo_embeddings"
    DEFAULT_CHECKPOINT="checkpoints/casanovo_v5_0_0.ckpt"
    CHECKPOINT_URL="https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt"
    ;;
  instanovo)
    EXTRACT_MODULE="instanovo_fm.eval.extract_instanovo_embeddings"
    DEFAULT_CHECKPOINT="checkpoints/instanovo_v1_2_0.ckpt"
    CHECKPOINT_URL="https://github.com/instadeepai/InstaNovo/releases/download/1.2.0/instanovo-v1.2.0.ckpt"
    ;;
  xuanjinovo)
    EXTRACT_MODULE="instanovo_fm.eval.extract_xuanjinovo_embeddings"
    DEFAULT_CHECKPOINT="checkpoints/XuanjiNovo_100M_massnet.ckpt"
    CHECKPOINT_URL="https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_100M_massnet.ckpt"
    ;;
  *)
    echo "ERROR: MODEL must be 'casanovo', 'instanovo' or 'xuanjinovo', got '$MODEL'" >&2
    exit 1
    ;;
esac
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"

# Build the per-model extractor args + an output-dir suffix so foundation/native and
# min_intensity variants don't overwrite each other.
EXTRA_EXTRACT_ARGS=()
OUTPUT_SUFFIX=""
# Only casanovo/xuanjinovo accept --preprocessing; instanovo runs the foundation pipeline only.
if [[ "$MODEL" == "casanovo" || "$MODEL" == "xuanjinovo" ]]; then
  EXTRA_EXTRACT_ARGS+=(--preprocessing "$PREPROCESSING")
  OUTPUT_SUFFIX="_${PREPROCESSING}"
fi
# --min_intensity is supported by all three extractors.
if [[ -n "${MIN_INTENSITY:-}" ]]; then
  EXTRA_EXTRACT_ARGS+=(--min_intensity "$MIN_INTENSITY")
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_mi${MIN_INTENSITY}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${EVAL_DATASET}/${MODEL}${OUTPUT_SUFFIX}}"

echo "============================================="
echo "Baseline UMAP + EVōC figures"
echo "  Model:           $MODEL"
echo "  Dataset:         $EVAL_DATASET"
echo "  Checkpoint:      $CHECKPOINT"
echo "  Pooling:         $POOLING   Device: $DEVICE"
echo "  Preprocessing:   $PREPROCESSING"
echo "  Eval config:     $EVAL_CONFIG_NAME"
echo "  Min intensity:   ${MIN_INTENSITY:-<extractor default>}"
echo "  Split:           $SPLIT"
echo "  Split parquet:   $SPLIT_PARQUET  (max_samples=$MAX_SAMPLES)"
echo "  max_mz:          $MAX_MZ"
echo "  Output dir:      $OUTPUT_DIR"
echo "============================================="

# ---------------------------------------------------------------------------- #
# macOS-only guard: the native EVōC/numba clustering (Step 2) pulls in a second
# OpenMP runtime alongside PyTorch's, which aborts/segfaults multi-threaded on
# Darwin. Force a single OpenMP runtime + single-threaded numba to avoid it.
# Skipped on Linux (AIchor) so the 100k EVōC clustering stays multi-threaded.
# ---------------------------------------------------------------------------- #
if [[ "$(uname)" == "Darwin" ]]; then
  export KMP_DUPLICATE_LIB_OK=TRUE
  export OMP_NUM_THREADS=1
  export NUMBA_NUM_THREADS=1
fi

# ---------------------------------------------------------------------------- #
# Step 1 — extract the <SPLIT>-split embeddings (+ metadata) to HDF5 + FAISS
# ---------------------------------------------------------------------------- #
echo ""
echo "--- Extracting $MODEL embeddings: $SPLIT (max_samples=$MAX_SAMPLES) ---"
uv run python -m "$EXTRACT_MODULE" \
  --parquet_path "$SPLIT_PARQUET" \
  --output_dir "${OUTPUT_DIR}/${SPLIT}" \
  --checkpoint_path "$CHECKPOINT" \
  --checkpoint_url "$CHECKPOINT_URL" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --max_samples "$MAX_SAMPLES" \
  --pooling "$POOLING" \
  --max_mz "$MAX_MZ" \
  "${EXTRA_EXTRACT_ARGS[@]+"${EXTRA_EXTRACT_ARGS[@]}"}"

# ---------------------------------------------------------------------------- #
# Step 2 — render UMAP + EVōC figures from the saved embeddings.
#   max_samples is bumped to MAX_SAMPLES (config default is 20k) to match the
#   original FM figures, and the EVōC cluster zoom-recolor panel is enabled.
# ---------------------------------------------------------------------------- #
echo ""
echo "[Tasks] umapvisualisationtask + evocclusteringtask (from saved embeddings)"
uv run python -m instanovo_fm.eval.run_tasks_from_embeddings \
  --embeddings_dir "${OUTPUT_DIR}/${SPLIT}" \
  --output_dir "$OUTPUT_DIR" \
  --tasks umapvisualisationtask evocclusteringtask \
  --eval_config_name "$EVAL_CONFIG_NAME" \
  --task_configs "{\"umapvisualisationtask\":{\"max_samples\":${MAX_SAMPLES}},\"evocclusteringtask\":{\"max_samples\":${MAX_SAMPLES},\"enable_cluster_zoom_recolor\":true}}"

echo ""
echo "[OK] $MODEL ($EVAL_DATASET) UMAP + EVōC figures complete — results in $OUTPUT_DIR"
