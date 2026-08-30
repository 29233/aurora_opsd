#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${AURORA_SOURCE_DIR:-/mnt/tbo/lvyf/AURORA/AURORA-main}"
SWIFT_ROOT="${MS_SWIFT_ROOT:-/mnt/tbo/lvyf/cate-pred-embedding}"
DATA_JSONL="${AURORA_DATA_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_smoke.jsonl}"
VAL_DATA_JSONL="${AURORA_VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_smoke.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_3b_smoke}"
MAX_STEPS="${MAX_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
REPORT_TO="${REPORT_TO:-tensorboard}"
# Evaluation uses ms-swift's native multi-card predict_with_generate path.
# Keep the generation cap explicit so each eval writes predictions to the
# trainer-managed predict.jsonl collection.
export AURORA_FREE_EVAL_MAX_TOKENS="${AURORA_FREE_EVAL_MAX_TOKENS:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1}"

export PYTHONPATH="${ROOT_DIR}:${SWIFT_ROOT}:${PYTHONPATH:-}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-2}"
export ENABLE_AUDIO_OUTPUT=0
export AURORA_DEBUG="${AURORA_DEBUG:-1}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"

python "${ROOT_DIR}/scripts/prepare_refavs_swift_sft.py" \
  --dataset-dir "${SOURCE_DIR}/dataset/REFAVS" \
  --metadata "${SOURCE_DIR}/dataset/REFAVS/metadata.csv" \
  --reasoning-json "${SOURCE_DIR%/*}/reason_cleaned.json" \
  --output "${DATA_JSONL}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-frames "${AURORA_NUM_SAM_FRAMES}"

python "${ROOT_DIR}/scripts/prepare_refavs_swift_sft.py" \
  --dataset-dir "${SOURCE_DIR}/dataset/REFAVS" \
  --metadata "${SOURCE_DIR}/dataset/REFAVS/metadata.csv" \
  --reasoning-json "${SOURCE_DIR%/*}/reason_cleaned.json" \
  --split val \
  --output "${VAL_DATA_JSONL}" \
  --max-samples "${VAL_MAX_SAMPLES}" \
  --num-frames "${AURORA_NUM_SAM_FRAMES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
swift sft \
  --model "${MODEL_PATH:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --dataset "${DATA_JSONL}" \
  --val_dataset "${VAL_DATA_JSONL}" \
  --new_special_tokens '[SEG]' \
  --train_type lora \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype float16 \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --max_length 4096 \
  --learning_rate 2e-5 \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --predict_with_generate true \
  --max_new_tokens "${AURORA_FREE_EVAL_MAX_TOKENS}" \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}"
