#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 FP8 optimized stack: TP8 + prefill CP8, FP8 KV cache,
# AITER DSA, tuned MoE/collectives, and EAGLE MTP.

: "${MODEL_PATH:?Set MODEL_PATH to the prepared zai-org/GLM-5.2-FP8 directory}"

IMAGE="${IMAGE:-sabreshao/sglang:glm_0831}"
CONTAINER_NAME="${CONTAINER_NAME:-glm52-prefill-optimized}"
CONTAINER_MODEL_PATH="${CONTAINER_MODEL_PATH:-/models/GLM-5.2-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

mount_args=(
  -v /boot:/boot:ro
  -v "${MODEL_PATH}:${CONTAINER_MODEL_PATH}:ro"
)
if [[ -e /dev/infiniband ]]; then
  mount_args+=(-v /dev/infiniband:/dev/infiniband)
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --entrypoint=/bin/bash \
  --init \
  --privileged \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=512G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${mount_args[@]}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_IB_TC=96 \
  -e NCCL_IB_FIFO_TC=184 \
  -e RDMAV_FORK_SAFE=1 \
  -e SGLANG_USE_AITER=1 \
  -e AITER_USE_FLYDSL_MOE_SORTING=1 \
  -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e SGLANG_DSA_A8W8=1 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=true \
  -e SGLANG_DSA_NO_PAD_HEADS=1 \
  -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e TORCHINDUCTOR_COMPILE_THREADS=128 \
  -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
  "${IMAGE}" \
  -lc "export PYTHONPATH=/opt/collective_opt/opt
export SGLANG_ENABLE_QUANTIZED_CP_MOE_AG=1
export SGLANG_QUANTIZED_CP_MOE_MIN_GLOBAL_ROWS=16384
export AITER_CONFIG_FMOE=/opt/aiter-configs/glm52_merged_tuned_fmoe.csv
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_DSA_A8W8=1
export SGLANG_DSA_NO_PAD_HEADS=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_FUSE_TOPK=true
export SGLANG_SANITIZE_NAN_LOGITS=1
export KV_DTYPE=fp8_e4m3

exec /opt/venv/bin/python -m sglang.launch_server \
  --model-path '${CONTAINER_MODEL_PATH}' \
  --served-model-name zai-org/GLM-5.2-FP8 \
  --trust-remote-code \
  --tp-size 8 \
  --context-length 1048576 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa \
  --dsa-prefill-backend aiter \
  --dsa-decode-backend aiter \
  --dsa-topk-backend sgl-kernel \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 131072 \
  --mem-fraction-static 0.80 \
  --max-running-requests 512 \
  --cuda-graph-max-bs-decode 64 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --attn-cp-size 8 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path '${CONTAINER_MODEL_PATH}' \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --speculative-moe-a2a-backend none \
  --speculative-moe-runner-backend auto \
  --enable-cache-report \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --decode-log-interval 1 \
  --random-seed 42 \
  --watchdog-timeout 1200 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host '${HOST}' \
  --port '${PORT}'"

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

echo "Optimized server is ready at http://127.0.0.1:${PORT}"
