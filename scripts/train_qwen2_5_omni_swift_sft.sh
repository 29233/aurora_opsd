#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Full-data entry point. Override these for a bounded validation run.
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-/mnt/tbo/lvyf/AURORA/AURORA-main/models/Qwen2___5-Omni-3B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_3b_sft}"
export AURORA_DATA_JSONL="${AURORA_DATA_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl}"
export AURORA_VAL_JSONL="${AURORA_VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val.jsonl}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export AURORA_DEBUG="${AURORA_DEBUG:-0}"

exec bash "${ROOT_DIR}/scripts/smoke_qwen2_5_omni_3b_swift_sft.sh"
