#!/usr/bin/env bash
# Prepare and run distributed retriever training on a rented NVIDIA GPU server.
#
# Put this file, train_retriever_st51.py, and
# requirements-retriever-st51.txt in the same directory on the server.
#
# Required environment variables:
#   DATASET_REPO=owner/dataset-repo
#   MODEL_REPO=owner/output-model-repo     # required only for train mode
#
# Authentication:
#   Run `hf auth login` once, or export HF_TOKEN from your provider's secret
#   manager. Do not paste a token into this script.
#
# Modes:
#   ./setup_retriever_server.sh prepare   # install, download, validate, cache data
#   ./setup_retriever_server.sh smoke     # 20-step local smoke test
#   ./setup_retriever_server.sh train     # full run + Hub checkpoints
#   ./setup_retriever_server.sh inspect   # environment/data checks only

set -Eeuo pipefail
IFS=$'\n\t'

MODE="${1:-prepare}"
case "${MODE}" in
  prepare|smoke|train|inspect) ;;
  *)
    echo "Usage: $0 {prepare|smoke|train|inspect}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$(pwd -P)"

# ---------------------------------------------------------------------------
# Configuration: override any of these with environment variables.
# ---------------------------------------------------------------------------

DATASET_REPO="${DATASET_REPO:-}"
MODEL_REPO="${MODEL_REPO:-}"
DATASET_REVISION="${DATASET_REVISION:-main}"
TRAIN_FILE="${TRAIN_FILE:-train.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-corpus.jsonl}"

WORK_ROOT="${WORK_ROOT:-${START_DIR}/retriever-work}"
DATA_DIR="${DATA_DIR:-${WORK_ROOT}/data}"
CACHE_DIR="${CACHE_DIR:-${WORK_ROOT}/retriever-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/runs/me5-base}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${WORK_ROOT}/hf-cache}"
VENV_DIR="${VENV_DIR:-${WORK_ROOT}/venv}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

TRAINER_SCRIPT="${TRAINER_SCRIPT:-${SCRIPT_DIR}/train_retriever_st51.py}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements-retriever-st51.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu124}" # also supported: cu118, cu126
SKIP_INSTALL="${SKIP_INSTALL:-0}"

NUM_GPUS="${NUM_GPUS:-2}"
BASE_MODEL="${BASE_MODEL:-intfloat/multilingual-e5-base}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
NEGATIVES="${NEGATIVES:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.01}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SEED="${SEED:-42}"
HUB_PRIVATE="${HUB_PRIVATE:-1}"
HUB_STRATEGY="${HUB_STRATEGY:-checkpoint}"
GATHER_ACROSS_DEVICES="${GATHER_ACROSS_DEVICES:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_FROM_HUB="${RESUME_FROM_HUB:-0}"

SMOKE_STEPS="${SMOKE_STEPS:-20}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
SMOKE_MINI_BATCH_SIZE="${SMOKE_MINI_BATCH_SIZE:-4}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-${WORK_ROOT}/runs/smoke}"
PUSH_SMOKE_TO_HUB="${PUSH_SMOKE_TO_HUB:-0}"
SMOKE_MODEL_REPO="${SMOKE_MODEL_REPO:-}"

if [[ -z "${DATASET_REPO}" ]]; then
  echo "DATASET_REPO is required, for example:" >&2
  echo "  export DATASET_REPO=your-name/retriever-data" >&2
  exit 2
fi
if [[ "${MODE}" == "train" && -z "${MODEL_REPO}" ]]; then
  echo "MODEL_REPO is required in train mode, for example:" >&2
  echo "  export MODEL_REPO=your-name/me5-base-retriever" >&2
  exit 2
fi
if [[ "${PUSH_SMOKE_TO_HUB}" == "1" && -z "${SMOKE_MODEL_REPO}" ]]; then
  echo "SMOKE_MODEL_REPO is required when PUSH_SMOKE_TO_HUB=1" >&2
  exit 2
fi
if [[ "${SAVE_STEPS}" -lt "${EVAL_STEPS}" ]] || (( SAVE_STEPS % EVAL_STEPS != 0 )); then
  echo "SAVE_STEPS must be a multiple of EVAL_STEPS" >&2
  exit 2
fi
case "${TORCH_CUDA_INDEX}" in
  cu118|cu124|cu126) ;;
  *)
    echo "TORCH_CUDA_INDEX must be cu118, cu124, or cu126 for PyTorch 2.6" >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_ROOT}" "${DATA_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}" \
  "${HF_CACHE_DIR}" "${LOG_DIR}"

# Keep HF_HOME unchanged so a token saved by `hf auth login` remains visible.
# Redirect the large download/dataset caches without relocating credentials.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_DIR}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Mode:              ${MODE}"
echo "Dataset:           ${DATASET_REPO}@${DATASET_REVISION}"
echo "Training file:     ${TRAIN_FILE}"
echo "Corpus file:       ${CORPUS_FILE}"
echo "Work directory:    ${WORK_ROOT}"
echo "CUDA wheel index:  ${TORCH_CUDA_INDEX}"
echo "Requested GPUs:    ${NUM_GPUS}"

# ---------------------------------------------------------------------------
# Python environment and pinned training stack.
# ---------------------------------------------------------------------------

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN} is not installed." >&2
    echo "Install Python 3.12 plus its venv package, or set PYTHON_BIN." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip setuptools wheel

  # Install the CUDA-enabled wheel explicitly. Installing the remaining
  # requirements afterwards will keep this build because it satisfies
  # torch>=2.6,<2.7.
  python -m pip install \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}" \
    "torch==2.6.0"

  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Requirements file not found: ${REQUIREMENTS_FILE}" >&2
    exit 1
  fi
  python -m pip install -r "${REQUIREMENTS_FILE}"
fi

python -m pip check

# ---------------------------------------------------------------------------
# Hardware, package, authentication, and cgroup diagnostics.
# ---------------------------------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found; select an NVIDIA/CUDA rental image." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader

REQUESTED_GPU_COUNT="${NUM_GPUS}" python - <<'PY'
import os
from importlib.metadata import version

import torch

expected = int(os.environ["REQUESTED_GPU_COUNT"])
count = torch.cuda.device_count()
print(f"torch={torch.__version__}")
print(f"transformers={version('transformers')}")
print(f"sentence-transformers={version('sentence-transformers')}")
print(f"accelerate={version('accelerate')}")
print(f"torch CUDA runtime={torch.version.cuda}; visible GPUs={count}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
if count < expected:
    raise SystemExit(f"requested {expected} GPUs but only {count} are visible")

for index in range(expected):
    torch.cuda.set_device(index)
    props = torch.cuda.get_device_properties(index)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(f"GPU {index} ({props.name}) does not support BF16")
    print(
        f"gpu {index}: {props.name}; "
        f"{props.total_memory / 2**30:.1f} GiB; BF16=yes"
    )
PY

if [[ -r /sys/fs/cgroup/memory.max ]]; then
  echo "cgroup memory.max:    $(< /sys/fs/cgroup/memory.max)"
fi
if [[ -r /sys/fs/cgroup/memory.current ]]; then
  echo "cgroup memory.current: $(< /sys/fs/cgroup/memory.current)"
fi
if [[ -r /sys/fs/cgroup/memory.events ]]; then
  echo "cgroup memory.events:"
  sed 's/^/  /' /sys/fs/cgroup/memory.events
fi
df -h "${WORK_ROOT}"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face authentication is required for private data and uploads." >&2
  echo "Run 'hf auth login' or export HF_TOKEN from a secret manager." >&2
  exit 1
fi
echo "Hugging Face authentication: OK"

# ---------------------------------------------------------------------------
# Download only the two required data files and validate their schemas.
# ---------------------------------------------------------------------------

hf download "${DATASET_REPO}" "${TRAIN_FILE}" "${CORPUS_FILE}" \
  --repo-type dataset \
  --revision "${DATASET_REVISION}" \
  --local-dir "${DATA_DIR}"

TRAIN_PATH="${DATA_DIR}/${TRAIN_FILE}"
CORPUS_PATH="${DATA_DIR}/${CORPUS_FILE}"

for required_file in "${TRAIN_PATH}" "${CORPUS_PATH}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Required file is missing or empty: ${required_file}" >&2
    exit 1
  fi
done

TRAIN_PATH="${TRAIN_PATH}" CORPUS_PATH="${CORPUS_PATH}" python - <<'PY'
import json
import os
from pathlib import Path


def first_nonempty_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
                if not isinstance(value, dict):
                    raise SystemExit(f"{path}:{line_number}: expected a JSON object")
                return value
    raise SystemExit(f"{path}: contains no records")


train_path = Path(os.environ["TRAIN_PATH"])
corpus_path = Path(os.environ["CORPUS_PATH"])
train = first_nonempty_json(train_path)
corpus = first_nonempty_json(corpus_path)

missing_train = {"query", "positives", "negatives"} - train.keys()
missing_corpus = {"id", "text"} - corpus.keys()
if missing_train:
    raise SystemExit(f"{train_path}: missing fields {sorted(missing_train)}")
if missing_corpus:
    raise SystemExit(f"{corpus_path}: missing fields {sorted(missing_corpus)}")
if not isinstance(train["positives"], list) or not isinstance(train["negatives"], list):
    raise SystemExit("train positives and negatives must be JSON lists")

print(f"Validated schemas: {train_path} and {corpus_path}")
PY

du -h "${TRAIN_PATH}" "${CORPUS_PATH}"

if [[ ! -f "${TRAINER_SCRIPT}" ]]; then
  echo "Trainer not found: ${TRAINER_SCRIPT}" >&2
  exit 1
fi

COMMON_ARGS=(
  --data "${TRAIN_PATH}"
  --corpus "${CORPUS_PATH}"
  --model "${BASE_MODEL}"
  --cache-dir "${CACHE_DIR}"
  --negatives "${NEGATIVES}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LEARNING_RATE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --seed "${SEED}"
)

if [[ "${GATHER_ACROSS_DEVICES}" == "1" ]]; then
  COMMON_ARGS+=(--gather-across-devices)
fi

case "${MODE}" in
  inspect)
    echo "Environment and data validation completed."
    ;;

  prepare)
    python "${TRAINER_SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --output "${OUTPUT_DIR}" \
      --prepare-data-only
    echo "Data cache prepared at ${CACHE_DIR}"
    ;;

  smoke)
    SMOKE_HUB_ARGS=()
    if [[ "${PUSH_SMOKE_TO_HUB}" == "1" ]]; then
      SMOKE_HUB_ARGS+=(
        --push-to-hub
        --hub-model-id "${SMOKE_MODEL_REPO}"
        --hub-strategy checkpoint
      )
      if [[ "${HUB_PRIVATE}" == "1" ]]; then
        SMOKE_HUB_ARGS+=(--hub-private)
      fi
    fi

    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
      "${TRAINER_SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --output "${SMOKE_OUTPUT}" \
      --max-steps "${SMOKE_STEPS}" \
      --batch-size "${SMOKE_BATCH_SIZE}" \
      --mini-batch-size "${SMOKE_MINI_BATCH_SIZE}" \
      --eval-steps "${SMOKE_STEPS}" \
      --save-steps "${SMOKE_STEPS}" \
      --logging-steps 1 \
      --save-total-limit 1 \
      "${SMOKE_HUB_ARGS[@]}"

    echo "Smoke test completed. TensorBoard logs: ${SMOKE_OUTPUT}/tensorboard"
    ;;

  train)
    HUB_ARGS=(
      --push-to-hub
      --hub-model-id "${MODEL_REPO}"
      --hub-strategy "${HUB_STRATEGY}"
    )
    if [[ "${HUB_PRIVATE}" == "1" ]]; then
      HUB_ARGS+=(--hub-private)
    fi

    if [[ "${RESUME_FROM_HUB}" == "1" ]]; then
      HUB_RESUME_DIR="${WORK_ROOT}/hub-resume"
      mkdir -p "${HUB_RESUME_DIR}"
      hf download "${MODEL_REPO}" \
        --repo-type model \
        --include 'last-checkpoint/*' \
        --local-dir "${HUB_RESUME_DIR}"
      RESUME_FROM_CHECKPOINT="${HUB_RESUME_DIR}/last-checkpoint"
      if [[ ! -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json" ]]; then
        echo "No resumable last-checkpoint found in ${MODEL_REPO}" >&2
        exit 1
      fi
    elif [[ -z "${RESUME_FROM_CHECKPOINT}" && "${AUTO_RESUME}" == "1" ]]; then
      RESUME_FROM_CHECKPOINT="$({
        find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null || true
      } | sort -V | tail -n 1)"
    fi

    RESUME_ARGS=()
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
      if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
        echo "Resume checkpoint not found: ${RESUME_FROM_CHECKPOINT}" >&2
        exit 1
      fi
      RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
      echo "Resuming from: ${RESUME_FROM_CHECKPOINT}"
    fi

    TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_LOG="${LOG_DIR}/train-${TIMESTAMP}.log"
    echo "Starting full training; combined log: ${RUN_LOG}"
    echo "For SSH resilience, run this mode inside tmux or a scheduler job."

    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
      "${TRAINER_SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --output "${OUTPUT_DIR}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --mini-batch-size "${MINI_BATCH_SIZE}" \
      --eval-steps "${EVAL_STEPS}" \
      --save-steps "${SAVE_STEPS}" \
      --logging-steps "${LOGGING_STEPS}" \
      --save-total-limit "${SAVE_TOTAL_LIMIT}" \
      "${HUB_ARGS[@]}" \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee "${RUN_LOG}"

    echo "Training completed."
    echo "Final local model: ${OUTPUT_DIR}/final"
    echo "Hub model: https://huggingface.co/${MODEL_REPO}"
    echo "TensorBoard logs: ${OUTPUT_DIR}/tensorboard"
    ;;
esac

python -m pip freeze > "${WORK_ROOT}/requirements.lock.txt"
echo "Resolved environment: ${WORK_ROOT}/requirements.lock.txt"
