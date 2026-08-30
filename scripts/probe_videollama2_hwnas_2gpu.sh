#!/usr/bin/env bash
set -euo pipefail

# Two-GPU smoke probe for a host where the shared mount is /mnt/hwnas-tbo.
# Override HWNAS_ROOT/AURORA_PROJECT_ROOT if the checkout is nested below a
# user-specific directory on the target machine.
HWNAS_ROOT="${HWNAS_ROOT:-/mnt/hwnas-tbo}"
PROJECT_ROOT="${AURORA_PROJECT_ROOT:-${HWNAS_ROOT}/lvyf/AURORA_from_scratch}"
VENV_PATH="${VENV_PATH:-${CONDA_PREFIX:-${VIRTUAL_ENV:-${HWNAS_ROOT}/lvyf/envs/aurora-ms-swift}}}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
SOURCE_DIR="${AURORA_SOURCE_DIR:-${HWNAS_ROOT}/lvyf/AURORA/AURORA-main}"
SWIFT_ROOT="${MS_SWIFT_ROOT:-${HWNAS_ROOT}/lvyf/cate-pred-embedding}"
MODEL_PATH="${MODEL_PATH:-${SOURCE_DIR}/models/VideoLLaMA2.1-7B-AV}"
SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-${HWNAS_ROOT}/lvyf/vmunet/best_model_and_code_extracted/pre_trained_weights/sam_vit_h_4b8939.pth}"

export CUDA_VISIBLE_DEVICES="4,5"
export NPROC_PER_NODE="2"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
if [[ ! -x "${PYTHON_BIN}" && "${PYTHON_BIN}" == "${VENV_PATH}/bin/python" && -x "${VENV_PATH}/bin/python3" ]]; then
  PYTHON_BIN="${VENV_PATH}/bin/python3"
fi
export PYTHON_BIN
if [[ -x "${VENV_PATH}/bin/swift" ]]; then
  export SWIFT_BIN="${VENV_PATH}/bin/swift"
  unset SWIFT_MODULE
else
  # A source checkout of ms-swift exposes the same CLI through this module.
  export SWIFT_MODULE="swift.cli.main"
fi
export AURORA_PROJECT_ROOT="${PROJECT_ROOT}"
export AURORA_SOURCE_DIR="${SOURCE_DIR}"
export MS_SWIFT_ROOT="${SWIFT_ROOT}"
export MODEL_PATH="${MODEL_PATH}"
export AURORA_SAM_CHECKPOINT="${SAM_CHECKPOINT}"
export VIDEOLLAMA2_SIGLIP_PATH="${VIDEOLLAMA2_SIGLIP_PATH:-${SOURCE_DIR}/models/siglip-so400m-patch14-384}"
export VIDEOLLAMA2_AUDIO_TOWER_PATH="${VIDEOLLAMA2_AUDIO_TOWER_PATH:-${MODEL_PATH}/audio_tower.bin}"
export VIDEOLLAMA2_NUM_FRAMES="${VIDEOLLAMA2_NUM_FRAMES:-2}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-2}"
export MAX_SAMPLES="${MAX_SAMPLES:-1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1}"
export EVAL_STEPS="${EVAL_STEPS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/videollama2_hwnas_2gpu_2f_probe}"
export AURORA_DATA_JSONL="${AURORA_DATA_JSONL:-${PROJECT_ROOT}/outputs/data/refavs_hwnas/refavs_train.jsonl}"
export AURORA_VAL_JSONL="${AURORA_VAL_JSONL:-${PROJECT_ROOT}/outputs/data/refavs_hwnas/refavs_val.jsonl}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Project directory does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "AURORA environment must provide python: ${VENV_PATH}" >&2
  echo "Override VENV_PATH if the environment is stored elsewhere." >&2
  exit 1
fi
if [[ -z "${SWIFT_MODULE:-}" && ! -x "${SWIFT_BIN}" ]]; then
  echo "Configured swift executable is not runnable: ${SWIFT_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "VideoLLaMA2 model directory does not contain config.json: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${AURORA_SAM_CHECKPOINT}" ]]; then
  echo "SAM checkpoint does not exist: ${AURORA_SAM_CHECKPOINT}" >&2
  exit 1
fi
for dataset in "${AURORA_DATA_JSONL}" "${AURORA_VAL_JSONL}"; do
  if [[ ! -f "${dataset}" ]]; then
    echo "Generated dataset does not exist: ${dataset}" >&2
    echo "Run ${PROJECT_ROOT}/scripts/prepare_refavs_hwnas_tbo_dataset.sh first." >&2
    exit 1
  fi
  if rg -n -m 1 '/mnt/tbo/' "${dataset}" >/dev/null; then
    echo "Dataset still contains paths from the old mount: ${dataset}" >&2
    exit 1
  fi
done

exec bash "${PROJECT_ROOT}/scripts/train_videollama2_swift_sft.sh"
