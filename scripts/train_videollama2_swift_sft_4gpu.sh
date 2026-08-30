#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Four-card V100/A100 entry point. Every value remains overridable by the
# environment so this wrapper can be used for both smoke tests and full SFT.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_DTYPE="${TORCH_DTYPE:-float16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json}"
export VIDEOLLAMA2_NUM_FRAMES="${VIDEOLLAMA2_NUM_FRAMES:-10}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
# Match the 8-card reference global batch: 4 GPUs x 2 accumulation = 8.
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/videollama2_4gpu_10f}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log}"
export LOG_FILE

echo "Training log: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

exec bash "${ROOT_DIR}/scripts/train_videollama2_swift_sft.sh"
