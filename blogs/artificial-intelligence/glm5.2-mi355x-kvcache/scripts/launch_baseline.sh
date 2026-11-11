#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 FP8 baseline: TP8, BF16 KV cache, TileLang DSA, no CP, no MTP.

: "${MODEL_PATH:?Set MODEL_PATH to the prepared zai-org/GLM-5.2-FP8 directory}"

IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618}"
CONTAINER_NAME="${CONTAINER_NAME:-glm52-prefill-baseline}"
CONTAINER_MODEL_PATH="${CONTAINER_MODEL_PATH:-/models/GLM-5.2-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

mount_args=(-v "${MODEL_PATH}:${CONTAINER_MODEL_PATH}:ro")

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network=host \
  --ipc=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=512G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${mount_args[@]}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${IMAGE}" \
  bash -lc "exec python3 -m sglang.launch_server \
    --model-path '${CONTAINER_MODEL_PATH}' \
    --served-model-name zai-org/GLM-5.2-FP8 \
    --host '${HOST}' \
    --port '${PORT}' \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 8}' \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --chunked-prefill-size 262144 \
    --watchdog-timeout 2400 \
    --kv-cache-dtype bf16 \
    --max-running-requests 200 \
    --cuda-graph-max-bs 200 \
    --context-length 1048576 \
    --enable-metrics \
    --enable-cache-report"

echo "Started ${CONTAINER_NAME}; waiting for http://127.0.0.1:${PORT}/health"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} exited during startup." >&2
    docker logs "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${CONTAINER_NAME}." >&2
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Baseline server is ready at http://127.0.0.1:${PORT}"
