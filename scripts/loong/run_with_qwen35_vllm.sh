#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LONGREFINER_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${LONGREFINER_LOONG_ENV_FILE:-${SCRIPT_DIR}/qwen35_longrefiner.env}"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--env-file" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    ENV_FILE="${ARGS[$((i + 1))]}"
  fi
done

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${LONGREFINER_PYTHON:-/opt/conda/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"
RUN_LOG="${LONGREFINER_RUN_LOG:-${LONGREFINER_ROOT}/logs/loong_qwen35_longrefiner.log}"
VLLM_LOG="${LONGREFINER_VLLM_LOG:-${LONGREFINER_ROOT}/logs/qwen35_vllm.log}"
INPUT_PATH="${LONGREFINER_INPUT_PATH:-/workspace/rag/Loong/loong_process.jsonl}"
OUTPUT_DIR="${LONGREFINER_OUTPUT_DIR:-${LONGREFINER_ROOT}/outputs/loong}"

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-${LLM_MODEL:-Qwen/Qwen3.5-27B}}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL}}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
HEALTH_TIMEOUT="${VLLM_HEALTH_TIMEOUT:-3600}"
TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
REASONING_PARSER="${VLLM_REASONING_PARSER-}"
MODEL_IMPL="${VLLM_MODEL_IMPL:-}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
VLLM_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
RUNNER_VISIBLE_DEVICES="${LONGREFINER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"

SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping vLLM server pid=${SERVER_PID}" | tee -a "${RUN_LOG}"
    kill -TERM -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}

wait_for_vllm() {
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import urllib.request

host = os.environ["VLLM_HOST"]
port = os.environ["VLLM_PORT"]
timeout_s = float(os.environ.get("VLLM_HEALTH_TIMEOUT", "3600"))
server_pid = os.environ.get("LONGREFINER_VLLM_SERVER_PID", "").strip()
url = f"http://{host}:{port}/health"
deadline = time.time() + timeout_s
last = ""

def server_exited(pid: str) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        state = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()[2]
    except Exception:
        return False
    return state == "Z"

while time.time() < deadline:
    if server_exited(server_pid):
        print(f"vLLM server exited before becoming ready: pid={server_pid}", file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 300:
                print(f"vLLM ready: {url}")
                sys.exit(0)
            last = f"status={response.status}"
    except Exception as exc:
        last = repr(exc)
    time.sleep(2)
print(f"vLLM did not become ready within {timeout_s:.0f}s: {last}", file=sys.stderr)
sys.exit(1)
PY
}

mkdir -p "$(dirname "${RUN_LOG}")" "$(dirname "${VLLM_LOG}")" "${OUTPUT_DIR}"
trap cleanup EXIT INT TERM

export VLLM_HOST="${HOST}"
export VLLM_PORT="${PORT}"
export VLLM_HEALTH_TIMEOUT="${HEALTH_TIMEOUT}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://${HOST}:${PORT}/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-${SERVED_MODEL_NAME}}"

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi
if [[ -n "${REASONING_PARSER}" ]]; then
  VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${MODEL_IMPL}" ]]; then
  VLLM_ARGS+=(--model-impl "${MODEL_IMPL}")
fi
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
  VLLM_ARGS+=(--default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}")
fi
if [[ -n "${CPU_OFFLOAD_GB}" ]]; then
  VLLM_ARGS+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi

echo "Starting Qwen3.5 vLLM ${MODEL} on ${HOST}:${PORT}" | tee "${RUN_LOG}"
if [[ -n "${VLLM_VISIBLE_DEVICES}" ]]; then
  echo "vLLM CUDA_VISIBLE_DEVICES=${VLLM_VISIBLE_DEVICES}" | tee -a "${RUN_LOG}"
  setsid env CUDA_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}" "${VLLM_PYTHON}" "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
else
  setsid "${VLLM_PYTHON}" "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
fi
SERVER_PID=$!
export LONGREFINER_VLLM_SERVER_PID="${SERVER_PID}"

if ! wait_for_vllm; then
  echo "vLLM failed to start. Last log lines:" | tee -a "${RUN_LOG}"
  tail -n 160 "${VLLM_LOG}" | tee -a "${RUN_LOG}" || true
  exit 1
fi

RUN_ARGS=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_loong_qwen35.py"
  --env-file "${ENV_FILE}"
  --input-path "${INPUT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  "$@"
)

if [[ -n "${RUNNER_VISIBLE_DEVICES}" ]]; then
  echo "LongRefiner runner CUDA_VISIBLE_DEVICES=${RUNNER_VISIBLE_DEVICES}" | tee -a "${RUN_LOG}"
  env CUDA_VISIBLE_DEVICES="${RUNNER_VISIBLE_DEVICES}" "${RUN_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"
else
  "${RUN_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"
fi
