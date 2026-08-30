#!/usr/bin/env bash
set -euo pipefail

# Submit-friendly 4-GPU entry point for the fixed VideoLLaMA2 SFT contract
# (audio_tower LoRA + trainable projectors + 10-frame audio/video alignment).
# Run with: bash scripts/submit_videollama2_4gpu.sh
# No detaching/redirection is performed; the scheduler owns this process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_DTYPE="${TORCH_DTYPE:-float16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json}"
export VIDEOLLAMA2_NUM_FRAMES="${VIDEOLLAMA2_NUM_FRAMES:-10}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
# Match the 8-card reference global batch: 4 GPUs x 2 accumulation = 8.
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/videollama2_4gpu_10f_fix}"
mkdir -p "${OUTPUT_DIR}"

exec bash "${ROOT_DIR}/scripts/train_videollama2_swift_sft.sh"
