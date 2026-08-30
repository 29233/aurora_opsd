#!/usr/bin/env bash
# AURORA OPSD training entry point (ms-swift 4.x GKD with teacher_prompt).
#
# Runs in the `swift46` conda env (ms-swift 4.5.2). Requires:
#   - dataset with a teacher_prompt column (scripts/prepare_refavs_opsd.py)
#   - swift_plugin/aurora_opsd.py (GKD trainer + AURORA mask supervision)
#
# Key configuration (see load.md analysis for rationale):
#   --lmbda 1        pure on-policy: every row uses the student's own rollout
#   --beta 0         forward KL (OPSD paper's ablation: reverse KL/JSD fail)
#   teacher          DYNAMIC self-distillation: no --teacher_model, the
#                    teacher is the student's own current weights conditioned
#                    on the privileged prompt (swift's OPSD mode)
#   mask loss        AURORA_OPSD_MASK_WEIGHT * (2*BCE + 0.5*Dice) added to the
#                    JSD loss by AuroraGKDTrainer (see swift_plugin/aurora_opsd.py)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${AURORA_SOURCE_DIR:-/mnt/tbo/lvyf/AURORA/AURORA-main}"

# Base model: the SFT Stage-1 checkpoint directory (LoRA adapter + SAM sidecar).
# The trainer loads it via --adapters so LoRA/embed/lm_head/text_hidden_fcs/
# mask_decoder all resume from Stage-1.
export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/opsd_class_r1}"

# OPSD dataset (teacher_prompt column injected by prepare_refavs_opsd.py)
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/refavs_full_opsd.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/refavs_val_opsd.jsonl}"

# OPSD / GKD hyperparameters
export LMBDA="${LMBDA:-1}"
export BETA="${BETA:-0}"
export TEMPERATURE="${TEMPERATURE:-1}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
export AURORA_OPSD_MASK_WEIGHT="${AURORA_OPSD_MASK_WEIGHT:-1.0}"

# Training schedule. The OPSD paper converges in ~100 steps on math; we budget
# 10x plus headroom (full dataset, 1 epoch) and rely on early checkpoints.
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"

# Env / infrastructure
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# A100 supports bf16 — preferred: avoids the fp32/fp16 mixed-dtype crash in
# transformers 5.x Qwen2.5-Omni generate (audio tower runs fp32; on fp16
# models q_proj then receives fp32 activations). Keep float16 for V100.
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-/mnt/tbo/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export ENABLE_AUDIO_OUTPUT=0
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

# PYTHONPATH: repo root for swift_plugin; no SWIFT_ROOT needed (pip-installed).
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- optional smoke-mode truncation --------------------------------------
if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  python - "${TRAIN_JSONL}" "${MAX_SAMPLES}" << 'EOF'
import json, sys
src, cap = sys.argv[1], int(sys.argv[2])
rows = []
with open(src) as f:
    for line in f:
        if len(rows) >= cap: break
        rows.append(line)
with open(src + '.smoke.jsonl', 'w') as f:
    f.writelines(rows)
print(f"smoke train set: {len(rows)} rows -> {src}.smoke.jsonl")
EOF
  export TRAIN_JSONL="${TRAIN_JSONL}.smoke.jsonl"
fi

if [[ "${VAL_MAX_SAMPLES}" != "-1" ]]; then
  python - "${VAL_JSONL}" "${VAL_MAX_SAMPLES}" << 'EOF'
import json, sys
src, cap = sys.argv[1], int(sys.argv[2])
rows = []
with open(src) as f:
    for line in f:
        if len(rows) >= cap: break
        rows.append(line)
with open(src + '.smoke.jsonl', 'w') as f:
    f.writelines(rows)
print(f"smoke val set: {len(rows)} rows -> {src}.smoke.jsonl")
EOF
  export VAL_JSONL="${VAL_JSONL}.smoke.jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift rlhf \
  --rlhf_type gkd \
  --model "${BASE_MODEL}" \
  --adapters "${STAGE1_ADAPTERS}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/aurora_opsd.py" \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --new_special_tokens '[SEG]' \
  --lmbda "${LMBDA}" \
  --beta "${BETA}" \
  --temperature "${TEMPERATURE}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --tuner_type lora \
  --ddp_static_graph true \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps "${MAX_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_length 4096 \
  --learning_rate "${LEARNING_RATE}" \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
