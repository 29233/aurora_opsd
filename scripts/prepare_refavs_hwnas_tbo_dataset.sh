#!/usr/bin/env bash
set -euo pipefail

# Rebuild all ms-swift JSONL files on the /mnt/hwnas-tbo mount.  Do not copy
# the JSONL files from another host: they contain absolute media paths.
HWNAS_ROOT="${HWNAS_ROOT:-/mnt/hwnas-tbo}"
PROJECT_ROOT="${AURORA_PROJECT_ROOT:-${HWNAS_ROOT}/lvyf/AURORA_from_scratch}"
SOURCE_DIR="${AURORA_SOURCE_DIR:-${HWNAS_ROOT}/lvyf/AURORA/AURORA-main}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-${VIRTUAL_ENV:-${HWNAS_ROOT}/lvyf/envs/aurora-ms-swift}}/bin/python}"
NUM_FRAMES="${NUM_FRAMES:-10}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PREPARE="${PROJECT_ROOT}/scripts/prepare_refavs_swift_sft.py"
DATASET_DIR="${SOURCE_DIR}/dataset/REFAVS"
METADATA="${DATASET_DIR}/metadata.csv"
REASONING_JSON="${SOURCE_DIR%/*}/reason_cleaned.json"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/data/refavs_hwnas}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter does not exist: ${PYTHON_BIN}" >&2
  exit 1
fi
for path in "${PREPARE}" "${METADATA}" "${REASONING_JSON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required dataset file does not exist: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
for split in train val test_s test_u test_n; do
  "${PYTHON_BIN}" "${PREPARE}" \
    --dataset-dir "${DATASET_DIR}" \
    --metadata "${METADATA}" \
    --reasoning-json "${REASONING_JSON}" \
    --split "${split}" \
    --max-samples "${MAX_SAMPLES}" \
    --num-frames "${NUM_FRAMES}" \
    --output "${OUTPUT_DIR}/refavs_${split}.jsonl"
done

# Fail early if an accidental old-host path remains in the generated files.
if rg -n '/mnt/tbo/' "${OUTPUT_DIR}"/*.jsonl >/dev/null; then
  echo "Generated dataset contains an unexpected mount path" >&2
  exit 1
fi
"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.glob('refavs_*.jsonl')):
    rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    if not rows:
        raise SystemExit(f'empty dataset: {path}')
    sample = rows[0]
    paths = []
    paths.extend(sample.get('audios', []))
    paths.extend(sample.get('sam_frame_paths', []))
    paths.extend(sample.get('mask_paths', []))
    paths.extend(sample.get('videos', [[]])[0])
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit(f'{path}: missing media path: {missing[0]}')
    print(f'{path.name}: {len(rows)} rows; paths verified')
PY
