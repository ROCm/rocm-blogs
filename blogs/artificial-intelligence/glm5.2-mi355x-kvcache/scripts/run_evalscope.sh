#!/usr/bin/env bash
set -euo pipefail

# Tested with EvalScope v0.5.2-737.

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/GLM-5.2-FP8 ./run_evalscope.sh <64k|128k|1m> <port> [host]

Examples:
  MODEL_PATH=/models/GLM-5.2-FP8 ./run_evalscope.sh 64k 30000
  MODEL_PATH=/models/GLM-5.2-FP8 ./run_evalscope.sh 128k 30001
  MODEL_PATH=/models/GLM-5.2-FP8 ./run_evalscope.sh 1m 30001

Optional overrides:
  CONCURRENCIES="1 2"        Override the profile's default concurrency list.
  RESULT_ROOT=/path/results  Override the output directory.
  TOKENIZER_PATH=/path       Override MODEL_PATH for tokenizer loading.
  PROMPTS_FACTOR=10          Requests per case = concurrency * this value.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

PROFILE="${1,,}"
PORT="$2"
HOST="${3:-127.0.0.1}"

: "${MODEL_PATH:?Set MODEL_PATH to the prepared zai-org/GLM-5.2-FP8 directory}"

MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVALSCOPE_BIN="${EVALSCOPE_BIN:-evalscope}"
PROMPTS_FACTOR="${PROMPTS_FACTOR:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$(pwd)/results/port-${PORT}/${PROFILE}/${RUN_ID}}"
URL="${URL:-http://${HOST}:${PORT}/v1/completions}"
EXTRA_ARGS="${EXTRA_ARGS:-{\"ignore_eos\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}}"

case "${PROFILE}" in
  64k)
    PREFIX_LENGTH=58982
    SUFFIX_LENGTH=6554
    DEFAULT_CONCURRENCIES="1 2 4 8 16 32 50"
    ;;
  128k)
    PREFIX_LENGTH=117964
    SUFFIX_LENGTH=13108
    DEFAULT_CONCURRENCIES="1 2 4 8 16 32 50"
    ;;
  1m)
    PREFIX_LENGTH=900000
    SUFFIX_LENGTH=100000
    DEFAULT_CONCURRENCIES="1"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

CONCURRENCIES="${CONCURRENCIES:-${DEFAULT_CONCURRENCIES}}"

if ! command -v "${EVALSCOPE_BIN}" >/dev/null 2>&1; then
  echo "EvalScope not found: ${EVALSCOPE_BIN}" >&2
  echo "Install the tested revision before running this script." >&2
  exit 1
fi

if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer path is not a directory: ${TOKENIZER_PATH}" >&2
  exit 1
fi

if ! curl -fsS "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "SGLang server is not healthy at http://${HOST}:${PORT}" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}/cases" "${RESULT_ROOT}/logs"
: > "${RESULT_ROOT}/logs/run_summary.log"

flush_cache() {
  curl -fsS -X POST "http://${HOST}:${PORT}/flush_cache" \
    -H "Content-Type: application/json" \
    -d '{}' >/dev/null
}

capture_metrics() {
  local destination="$1"
  curl -fsS "http://${HOST}:${PORT}/metrics" -o "${destination}"
}

calculate_cache_delta() {
  local before_file="$1"
  local after_file="$2"
  local output_file="$3"

  python3 - "${before_file}" "${after_file}" "${output_file}" <<'PY'
import re
import sys
from pathlib import Path

before_path, after_path, output_path = map(Path, sys.argv[1:4])

def metric(path: Path, name: str, default: float | None = None) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"^" + re.escape(name) + r"\{[^}]*\}\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        if default is not None:
            return default
        raise RuntimeError(f"Metric not found: {name}")
    return float(match.group(1))

prompt_delta = (
    metric(after_path, "sglang:prompt_tokens_total")
    - metric(before_path, "sglang:prompt_tokens_total", default=0.0)
)
cached_delta = (
    metric(after_path, "sglang:cached_tokens_total")
    - metric(before_path, "sglang:cached_tokens_total", default=0.0)
)
cache_hit = cached_delta / prompt_delta if prompt_delta else float("nan")
result = (
    f"prompt_delta={prompt_delta:.0f} "
    f"cached_delta={cached_delta:.0f} "
    f"delta_cache_hit={cache_hit:.6f}\n"
)
output_path.write_text(result, encoding="utf-8")
print(result, end="")
PY
}

run_case() {
  local concurrency="$1"
  local number=$((concurrency * PROMPTS_FACTOR))
  local case_id="glm52-${PROFILE}-prefill-c$(printf '%03d' "${concurrency}")-n$(printf '%04d' "${number}")"
  local case_dir="${RESULT_ROOT}/cases/${case_id}"
  local log_file="${RESULT_ROOT}/logs/${case_id}.log"
  local metrics_before="${RESULT_ROOT}/logs/${case_id}.metrics-before.txt"
  local metrics_after="${RESULT_ROOT}/logs/${case_id}.metrics-after.txt"
  local cache_delta="${RESULT_ROOT}/logs/${case_id}.delta-cache-hit.txt"

  mkdir -p "${case_dir}"

  echo "===== START ${case_id} prefix=${PREFIX_LENGTH} suffix=${SUFFIX_LENGTH} =====" |
    tee -a "${RESULT_ROOT}/logs/run_summary.log"

  flush_cache
  capture_metrics "${metrics_before}"

  "${EVALSCOPE_BIN}" perf \
    --api openai \
    --url "${URL}" \
    --model "${MODEL}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --dataset random \
    --prefix-length "${PREFIX_LENGTH}" \
    --min-prompt-length "${SUFFIX_LENGTH}" \
    --max-prompt-length "${SUFFIX_LENGTH}" \
    --parallel "${concurrency}" \
    --number "${number}" \
    --max-tokens "${OUTPUT_LEN}" \
    --min-tokens "${OUTPUT_LEN}" \
    --temperature 0 \
    --no-stream \
    --warmup-num 1 \
    --seed "${SEED}" \
    --extra-args "${EXTRA_ARGS}" \
    --outputs-dir "${case_dir}" \
    --name "${case_id}" \
    --no-timestamp 2>&1 | tee "${log_file}"

  capture_metrics "${metrics_after}"
  calculate_cache_delta "${metrics_before}" "${metrics_after}" "${cache_delta}" |
    tee -a "${RESULT_ROOT}/logs/run_summary.log"

  echo "===== END ${case_id} =====" |
    tee -a "${RESULT_ROOT}/logs/run_summary.log"
}

for concurrency in ${CONCURRENCIES}; do
  run_case "${concurrency}"
done

echo "All ${PROFILE} EvalScope cases finished. Results: ${RESULT_ROOT}"
