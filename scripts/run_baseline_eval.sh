#!/usr/bin/env bash
# =============================================================================
# Embedding Evaluation (Casanovo / InstaNovo / XuanjiNovo baselines + FM + FM de novo)
#
# Usage (env-var driven):
#   MODEL=instanovo EVAL_DATASET=lcfm bash scripts/run_baseline_eval.sh                   # AIchor baseline
#   MODEL=fm CHECKPOINT=s3://.../model_best.ckpt EVAL_DATASET=lcfm bash scripts/run_baseline_eval.sh   # FM self-eval
#   MODEL=fm_de_novo CHECKPOINT=s3://.../model_best.ckpt EVAL_DATASET=lcfm bash scripts/run_baseline_eval.sh   # downstream de novo encoder
#
# Parameters (all overridable via environment):
#   MODEL             casanovo | instanovo | xuanjinovo | fm | fm_de_novo   (required)
#                     fm_de_novo evaluates the encoder of a DownstreamDeNovo checkpoint
#                     (FM encoder + InstaNovo decoder) via the same extract+probe path as
#                     the baselines. It has no canonical checkpoint, so CHECKPOINT is required.
#   EVAL_DATASET      dataset-config name               (default: lcfm)
#                     The literal filename of src/instanovo_fm/configs/dataset/<EVAL_DATASET>.yaml
#                     (e.g. lcfm, mcfm, hcfm)
#   SPLIT             valid | test | train                (default: valid)
#                     Which split the SINGLE-split tasks (duplicateretrievaltask,
#                     umapvisualisationtask, …) run on. The linear-probe train/val/test
#                     splits are unaffected — they always use their own parquets.
#                     Embeddings are extracted into ${OUTPUT_DIR}/<SPLIT>. Also forwarded to
#                     the FM (MODEL=fm) branch as evaluation.split.
#   DUP_MAX_SAMPLES   <SPLIT> pool size for single-split tasks (default: 200000)
#                     MUST equal the FM run's evaluation.max_samples to be comparable.
#   POOLING           mean_peaks (default) | native summary token (casanovo=cls, instanovo=latent)
#   DEVICE            cuda | cpu                         (default: cuda)
#   BATCH_SIZE        inference batch size              (default: 256)
#   NUM_WORKERS       dataloader workers                (default: 4)
#   EVAL_CONFIG_NAME  evaluation config to source        (default: default)
#   CHECKPOINT        override checkpoint path           (default: per-model canonical ckpt)
#                     May be a local path, an http(s) URL fallback target, or an
#                     s3://bucket/key URI (resolved via S3FileHandler — requires S3 env vars
#                     AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, e.g. from .env).
#                     S3 checkpoints are cached under checkpoints/ so they download only once.
#                     e.g. CHECKPOINT=s3://my-bucket/instanovo_v1_2_0.ckpt MODEL=instanovo bash scripts/run_baseline_eval.sh
#   TRAIN_PARQUET / VALID_PARQUET / TEST_PARQUET  override per-split parquet/glob
#   OUTPUT_DIR        override output directory          (default: .../baselines/<dataset>/<model>)
# =============================================================================

set -euo pipefail

# Parameters
MODEL="${MODEL:?MODEL must be set to 'casanovo', 'instanovo', 'xuanjinovo', 'fm' or 'fm_de_novo'}"
TASKS="${TASKS:-linearprobetask}"
EVAL_DATASET="${EVAL_DATASET:-lcfm}"
SPLIT="${SPLIT:-valid}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DUP_MAX_SAMPLES="${DUP_MAX_SAMPLES:-200000}"

case "$SPLIT" in
  valid|test|train) ;;
  *) echo "ERROR: SPLIT must be 'valid', 'test' or 'train', got '$SPLIT'" >&2; exit 1 ;;
esac

# FM self-evaluation short-circuit.
if [[ "$MODEL" == "fm" ]]; then
  CHECKPOINT="${CHECKPOINT:?CHECKPOINT must be set to a local path or s3:// URI of the FM checkpoint when MODEL=fm}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  OUTPUT_DIR="${OUTPUT_DIR:-embed_eval_results/fm}"

  echo "============================================="
  echo "FM embedding eval"
  echo "  Checkpoint:   $CHECKPOINT"
  echo "  Dataset:      $EVAL_DATASET"
  echo "  Split:        $SPLIT"
  echo "  Max samples:  $DUP_MAX_SAMPLES   Batch: $BATCH_SIZE   Workers: $NUM_WORKERS"
  echo "  Output dir:   $OUTPUT_DIR"
  echo "============================================="

  uv run python -m instanovo_fm.eval.embed_evaluation \
    --config-name foundational \
    "dataset=${EVAL_DATASET}" \
    evaluation.checkpoint_path="$CHECKPOINT" \
    evaluation.output_dir="$OUTPUT_DIR" \
    evaluation.split="$SPLIT" \
    evaluation.batch_size="$BATCH_SIZE" \
    evaluation.max_samples="$DUP_MAX_SAMPLES" \
    evaluation.tasks_to_run=["$TASKS"] \
    evaluation.task_configs.linearprobetask.use_project_split=false \
    mlflow_enabled=False \
    num_workers="$NUM_WORKERS"

  echo ""
  echo "[OK] FM ($EVAL_DATASET) eval complete — results in $OUTPUT_DIR"
  exit 0
fi

POOLING="${POOLING:-mean_peaks}"
PREPROCESSING="${PREPROCESSING:-foundation}" # Foundation or native preprocessing pipeline
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-default}"
TASK_CONFIGS="${TASK_CONFIGS:-}"

OUTPUT_BASE="embed_eval_results"

IFS='|' read -r CFG_TRAIN CFG_VALID CFG_TEST CFG_DATA CFG_MAX_MZ < <(
  uv run python -c "
from instanovo_fm.eval.run_tasks_from_embeddings import _load_dataset_paths, _load_foundation_max_mz
p = _load_dataset_paths('${EVAL_DATASET}')
print(p.get('train_path') or '', p.get('valid_path') or '', p.get('test_path') or '', p.get('data_path') or '', _load_foundation_max_mz(), sep='|')" | tail -n 1
)
TRAIN_PARQUET="${TRAIN_PARQUET:-$CFG_TRAIN}"
VALID_PARQUET="${VALID_PARQUET:-$CFG_VALID}"
TEST_PARQUET="${TEST_PARQUET:-$CFG_TEST}"
DATA_PATH="${DATA_PATH:-$CFG_DATA}" # Single-path alternative
MAX_MZ="${MAX_MZ:-$CFG_MAX_MZ}"

if [[ -z "$TRAIN_PARQUET" || -z "$VALID_PARQUET" || -z "$TEST_PARQUET" ]]; then
  if [[ -z "$DATA_PATH" ]]; then
    echo "ERROR: could not resolve train/valid/test paths (or a single data_path) for EVAL_DATASET='${EVAL_DATASET}'." >&2
    echo "       Expected src/instanovo_fm/configs/dataset/${EVAL_DATASET}.yaml with train_path/valid_path/test_path," >&2
    echo "       OR a single data_path to split. got: train='${TRAIN_PARQUET}' valid='${VALID_PARQUET}' test='${TEST_PARQUET}' data_path='${DATA_PATH}'" >&2
    exit 1
  fi
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
  fm_de_novo)
    EXTRACT_MODULE="instanovo_fm.eval.extract_fm_de_novo_embeddings"
    DEFAULT_CHECKPOINT=""
    CHECKPOINT_URL=""
    ;;
  *)
    echo "ERROR: MODEL must be 'casanovo', 'instanovo', 'xuanjinovo', 'fm' or 'fm_de_novo', got '$MODEL'" >&2
    exit 1
    ;;
esac
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"

# fm_de_novo (and any future checkpoint-only model) has no canonical checkpoint to fall back to.
if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: MODEL=$MODEL has no default checkpoint; set CHECKPOINT to a local path or s3:// URI of the trained checkpoint." >&2
  exit 1
fi

# Only casanovo/xuanjinovo accept --preprocessing; tag their default output dir with the mode so
# foundation/native comparison runs don't overwrite each other.
EXTRA_EXTRACT_ARGS=()
OUTPUT_SUFFIX=""
if [[ "$MODEL" == "casanovo" || "$MODEL" == "xuanjinovo" ]]; then
  EXTRA_EXTRACT_ARGS+=(--preprocessing "$PREPROCESSING")
  OUTPUT_SUFFIX="_${PREPROCESSING}"
fi
if [[ -n "${MIN_INTENSITY:-}" ]]; then
  EXTRA_EXTRACT_ARGS+=(--min_intensity "$MIN_INTENSITY")
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_mi${MIN_INTENSITY}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${MODEL}${OUTPUT_SUFFIX}}"

# ---------------------------------------------------------------------------- #
# Single-path mode: deterministically split one data_path into sequence-disjoint
# train/valid/test partitions (reproducible: sorted unique sequences + fixed seed).
# The three globs then flow through the unchanged extract + probe steps below.
# ---------------------------------------------------------------------------- #
if [[ -z "$TRAIN_PARQUET" && -z "$VALID_PARQUET" && -z "$TEST_PARQUET" && -n "$DATA_PATH" ]]; then
  echo "--- Splitting single data_path into sequence-disjoint train/valid/test: $DATA_PATH ---"
  IFS=$'\t' read -r TRAIN_PARQUET VALID_PARQUET TEST_PARQUET < <(
    uv run python -m instanovo_fm.eval.split_data_path \
      --data_path "$DATA_PATH" \
      --output_dir "${OUTPUT_DIR}/_splits" \
      --seed "${SPLIT_SEED:-42}" \
      --group_key "${SPLIT_GROUP_KEY:-sequence}" | tail -n 1
  )
  if [[ -z "$TRAIN_PARQUET" || -z "$VALID_PARQUET" || -z "$TEST_PARQUET" ]]; then
    echo "ERROR: split_data_path did not return three paths for DATA_PATH='${DATA_PATH}'." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------- #
# Resolve the single-split pool (duplicate retrieval / umap) from SPLIT.
# The linear-probe train/val/test splits below are separate and unaffected.
# ---------------------------------------------------------------------------- #
case "$SPLIT" in
  train) POOL_PARQUET="$TRAIN_PARQUET" ;;
  valid) POOL_PARQUET="$VALID_PARQUET" ;;
  test)  POOL_PARQUET="$TEST_PARQUET" ;;
esac

# ---------------------------------------------------------------------------- #
# Read the linear-probe per-split caps from the SAME eval config the FM uses, so
# the baseline probe splits have identical sizes (no drift / hand-tuned numbers).
# ---------------------------------------------------------------------------- #
read -r TRAIN_SAMPLES VAL_SAMPLES TEST_SAMPLES < <(
  uv run python -c "
from instanovo_fm.eval.run_tasks_from_embeddings import _load_eval_task_configs
c = _load_eval_task_configs('${EVAL_CONFIG_NAME}').get('linearprobetask', {})
print(c.get('train_samples', 100000), c.get('val_samples', 10000), c.get('test_samples', 10000))
" | tail -n 1
)

# Fail fast (before any GPU work) if the caps didn't parse as integers.
int_re='^[0-9]+$'
if ! [[ "$TRAIN_SAMPLES" =~ $int_re && "$VAL_SAMPLES" =~ $int_re && "$TEST_SAMPLES" =~ $int_re ]]; then
  echo "ERROR: failed to read integer probe caps from eval config '${EVAL_CONFIG_NAME}'." >&2
  echo "       got: train='${TRAIN_SAMPLES}' val='${VAL_SAMPLES}' test='${TEST_SAMPLES}'" >&2
  echo "       (the config-read subprocess likely printed library log noise to stdout)" >&2
  exit 1
fi

echo "============================================="
echo "Baseline embedding eval (multi-split probe)"
echo "  Model:           $MODEL"
echo "  Dataset:         $EVAL_DATASET"
echo "  Checkpoint:      $CHECKPOINT"
echo "  Pooling:         $POOLING   Device: $DEVICE"
echo "  Eval config:     $EVAL_CONFIG_NAME"
echo "  max_mz:          $MAX_MZ"
echo "  Single-split:    $SPLIT"
echo "  Dup/umap pool:   $POOL_PARQUET  (max_samples=$DUP_MAX_SAMPLES)"
echo "  Probe train:     $TRAIN_PARQUET  (max_samples=$TRAIN_SAMPLES)"
echo "  Probe val:       $VALID_PARQUET  (max_samples=$VAL_SAMPLES)"
echo "  Probe test:      $TEST_PARQUET   (max_samples=$TEST_SAMPLES)"
echo "  Output dir:      $OUTPUT_DIR"
echo "============================================="

# ---------------------------------------------------------------------------- #
# Helper: extract one split into a subdir of OUTPUT_DIR
# ---------------------------------------------------------------------------- #
extract_split() {
  local parquet="$1" max_samples="$2" subdir="$3"
  echo ""
  echo "--- Extracting $MODEL embeddings: $subdir (max_samples=$max_samples) ---"
  uv run python -m "$EXTRACT_MODULE" \
    --parquet_path "$parquet" \
    --output_dir "${OUTPUT_DIR}/${subdir}" \
    --checkpoint_path "$CHECKPOINT" \
    --checkpoint_url "$CHECKPOINT_URL" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS" \
    --max_samples "$max_samples" \
    --pooling "$POOLING" \
    --max_mz "$MAX_MZ" \
    "${EXTRA_EXTRACT_ARGS[@]+"${EXTRA_EXTRACT_ARGS[@]}"}"
}

# ---------------------------------------------------------------------------- #
# Step 1 — extract the single-split pool (duplicate retrieval / umap) + the probe
# train/val/test splits. The pool split is SPLIT (default valid); DUP_MAX_SAMPLES
# just caps at its size.
# ---------------------------------------------------------------------------- #
extract_split "$POOL_PARQUET"  "$DUP_MAX_SAMPLES" "$SPLIT"
# Probe splits are only needed by linearprobetask — skip the (up to 100k) train/val/test
# extractions when it isn't requested (e.g. a dup-retrieval / umap-only run).
if [[ " $TASKS " == *" linearprobetask "* ]]; then
  extract_split "$TRAIN_PARQUET" "$TRAIN_SAMPLES" "probe_train"
  extract_split "$VALID_PARQUET" "$VAL_SAMPLES"   "probe_val"
  extract_split "$TEST_PARQUET"  "$TEST_SAMPLES"  "probe_test"
fi

# ---------------------------------------------------------------------------- #
# Step 2 — run tasks with the canonical (FM-matched) config.
#   duplicate retrieval / umap -> ${SPLIT} pool;  linear probe -> train/val/test (pre_filtered)
# ---------------------------------------------------------------------------- #
echo ""
echo "[Tasks] $TASKS"
# Only wire probe split dirs when the linear probe runs (they aren't extracted otherwise).
PROBE_ARGS=()
if [[ " $TASKS " == *" linearprobetask "* ]]; then
  PROBE_ARGS+=(--probe_split_dirs "train=${OUTPUT_DIR}/probe_train,val=${OUTPUT_DIR}/probe_val,test=${OUTPUT_DIR}/probe_test")
fi
if [[ -n "$TASK_CONFIGS" ]]; then
  PROBE_ARGS+=(--task_configs "$TASK_CONFIGS")
fi
# $TASKS is intentionally unquoted so multiple space-separated task names expand as
# separate argv entries (--tasks uses argparse nargs="+").
uv run python -m instanovo_fm.eval.run_tasks_from_embeddings \
  --embeddings_dir "${OUTPUT_DIR}/${SPLIT}" \
  --output_dir "$OUTPUT_DIR" \
  --tasks $TASKS \
  --eval_config_name "$EVAL_CONFIG_NAME" \
  "${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}"

echo ""
echo "[OK] $MODEL ($EVAL_DATASET) baseline eval complete — results in $OUTPUT_DIR"
