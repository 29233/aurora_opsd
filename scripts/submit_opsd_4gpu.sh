#!/usr/bin/env bash
# AURORA OPSD 4-GPU training entry (server submission style).
#
# Runs in the `swift46` conda env. Plain `bash scripts/submit_opsd_4gpu.sh` —
# no setsid / nohup / output redirection; the scheduler owns the process and
# the wrapper tees its own log to OUTPUT_DIR/train_<timestamp>.log.
#
# Multi-GPU note: the DDP-symmetric mask branch (e67af69) is REQUIRED —
# without it ranks deadlock as soon as one rank's rollout lacks [SEG].
# eval every 1000 steps, checkpoint every 500.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/opsd_class_r1_4gpu}"

export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"

bash "${ROOT_DIR}/scripts/train_opsd_swift.sh"
