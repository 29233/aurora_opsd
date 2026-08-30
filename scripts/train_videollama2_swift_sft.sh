#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${AURORA_SOURCE_DIR:-/mnt/tbo/lvyf/AURORA/AURORA-main}"
SWIFT_ROOT="${MS_SWIFT_ROOT:-/mnt/tbo/lvyf/cate-pred-embedding}"
MODEL_PATH="${MODEL_PATH:-${SOURCE_DIR}/models/VideoLLaMA2.1-7B-AV}"
DATA_JSONL="${AURORA_DATA_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl}"
VAL_JSONL="${AURORA_VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/videollama2_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_MODULE="${SWIFT_MODULE:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
DATASET_SPEC="${DATA_JSONL}"
VAL_DATASET_SPEC="${VAL_JSONL}"
if [[ ! -f "${DATA_JSONL}" && -f "${ROOT_DIR}/outputs/refavs_full.jsonl" ]]; then
  echo "Warning: new data path is missing; using legacy dataset ${ROOT_DIR}/outputs/refavs_full.jsonl" >&2
  DATA_JSONL="${ROOT_DIR}/outputs/refavs_full.jsonl"
fi
if [[ ! -f "${VAL_JSONL}" && -f "${ROOT_DIR}/outputs/refavs_val.jsonl" ]]; then
  echo "Warning: new validation path is missing; using legacy dataset ${ROOT_DIR}/outputs/refavs_val.jsonl" >&2
  VAL_JSONL="${ROOT_DIR}/outputs/refavs_val.jsonl"
fi
DATASET_SPEC="${DATA_JSONL}"
VAL_DATASET_SPEC="${VAL_JSONL}"
if (( MAX_SAMPLES > 0 )); then
  DATASET_SPEC="${DATA_JSONL}#${MAX_SAMPLES}"
fi
if (( VAL_MAX_SAMPLES > 0 )); then
  VAL_DATASET_SPEC="${VAL_JSONL}#${VAL_MAX_SAMPLES}"
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/VideoLLaMA2av:${SWIFT_ROOT}:${PYTHONPATH:-}"
export VIDEOLLAMA2_MODEL_PATH="${MODEL_PATH}"
export VIDEOLLAMA2_SIGLIP_PATH="${VIDEOLLAMA2_SIGLIP_PATH:-${SOURCE_DIR}/models/siglip-so400m-patch14-384}"
export VIDEOLLAMA2_AUDIO_TOWER_PATH="${VIDEOLLAMA2_AUDIO_TOWER_PATH:-${MODEL_PATH}/audio_tower.bin}"
export VIDEOLLAMA2_NUM_FRAMES="${VIDEOLLAMA2_NUM_FRAMES:-10}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth}"

if [[ -n "${SWIFT_MODULE}" ]]; then
  SWIFT_CMD=("${PYTHON_BIN}" -m "${SWIFT_MODULE}")
elif [[ -x "${SWIFT_BIN}" ]] || command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
  SWIFT_CMD=("${SWIFT_BIN}")
elif [[ -x "${PYTHON_BIN}" && -d "${SWIFT_ROOT}/swift" ]]; then
  # Source checkouts of ms-swift do not always install the console script.
  SWIFT_CMD=("${PYTHON_BIN}" -m swift.cli.main)
else
  echo "Cannot find a runnable ms-swift CLI. Set SWIFT_BIN or SWIFT_MODULE." >&2
  exit 1
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

exec "${SWIFT_CMD[@]}" sft \
  --model "${MODEL_PATH}" \
  --model_type aurora_videollama2_qwen2 \
  --template aurora_videollama2 \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_videollama2.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --dataset "${DATASET_SPEC}" \
  --val_dataset "${VAL_DATASET_SPEC}" \
  --new_special_tokens '[SEG]' \
  --train_type lora \
  --target_regex '^(model\.layers\.\d+\.self_attn\.(q_proj|v_proj)|model\.audio_tower\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj))$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder mm_projector mm_projector_a \
  --freeze_vit true \
  --freeze_aligner false \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS:-500}" \
  --predict_with_generate true \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --gradient_checkpointing true \
  "${DEEPSPEED_ARGS[@]}" \
  --report_to "${REPORT_TO:-tensorboard}" \
  --output_dir "${OUTPUT_DIR}"
