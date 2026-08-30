#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Four-card Qwen2.5-Omni-7B SFT entry point. Two accumulation steps keep the
# effective global batch equivalent to the reference 8-card configuration.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-7B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_4gpu_10f}"
export AURORA_SOURCE_DIR="${AURORA_SOURCE_DIR:-/mnt/tbo/lvyf/AURORA/AURORA-main}"
export MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/mnt/tbo/lvyf/cate-pred-embedding}"
export AURORA_DATA_JSONL="${AURORA_DATA_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl}"
export AURORA_VAL_JSONL="${AURORA_VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val.jsonl}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log}"
echo "Training log: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

exec bash "${ROOT_DIR}/scripts/train_qwen2_5_omni_swift_sft.sh"
