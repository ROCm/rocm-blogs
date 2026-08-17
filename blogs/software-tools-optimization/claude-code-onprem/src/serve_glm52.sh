#!/usr/bin/env bash
#
# Bring up GLM-5.2-FP8 with SGLang and an OpenAI-compatible endpoint on your node.
# This is the stock-image serve that works out of the box on MI355X (gfx950).
# glm-setup can't deploy this for you (it needs the weights on the box), but if
# you drop this on the node and point MODEL_PATH at your local checkpoint, it
# will start the server that the router and Claude Code talk to.
# Author: adlashab <adlashab@amd.com>
#
# Prereqs on the node:
#   - the SGLang image below is pulled locally (docker images | grep sgl-dev)
#   - the GLM-5.2-FP8 checkpoint is on LOCAL NVMe (network filesystems are far
#     too slow to load from)
#   - idle gfx950 GPUs (check: rocm-smi --showuse, all ~0%, no foreign process)
#   - docker access (plain docker, or passwordless sudo docker; auto-detected)
#
# Usage:
#   MODEL_PATH=/path/to/GLM-5.2-FP8 ./serve_glm52.sh          # start (TP8, :31090)
#   ./serve_glm52.sh stop                                     # stop + remove
#   MODEL_PATH=... PORT=31090 TP_SIZE=8 ./serve_glm52.sh
#
# Once it prints READY:
#   curl -s http://127.0.0.1:31090/v1/models

set -u

# ---- parameters (MODEL_PATH is the one you must set) ----
MODEL_PATH=${MODEL_PATH:-}
PORT=${PORT:-31090}
TP_SIZE=${TP_SIZE:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
HOST=${HOST:-127.0.0.1}
SERVED_NAME=${SERVED_NAME:-glm-5.2-fp8}
IMAGE=${IMAGE:-rocm/sgl-dev:v0.5.15.post1-rocm720-mi35x-20260714}
CONTAINER=${CONTAINER:-glm52_serve}
READY_TIMEOUT=${READY_TIMEOUT:-900}
MEM_FRACTION=${MEM_FRACTION:-0.90}

if [ "${1:-}" != "stop" ] && [ -z "$MODEL_PATH" ]; then
  echo "ERROR: set MODEL_PATH to your local GLM-5.2-FP8 checkpoint, e.g." >&2
  echo "  MODEL_PATH=/data/you/models/GLM-5.2-FP8 $0" >&2
  exit 2
fi

BASE_URL="http://${HOST}:${PORT}"
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/glm52_serve_logs}
LOG=$LOG_DIR/sglang_glm52_tp${TP_SIZE}_p${PORT}_$(date +%Y%m%d_%H%M%S).log

# Some nodes don't put the login user in the docker group, so a direct login
# can't reach the daemon socket. Pick the invocation that works: plain docker if
# the socket answers, else sudo docker. Force it with DOCKER="sudo docker".
if [ -z "${DOCKER:-}" ]; then
  if docker ps >/dev/null 2>&1; then DOCKER="docker"
  elif sudo -n docker ps >/dev/null 2>&1; then DOCKER="sudo docker"
  else
    echo "ERROR: can't reach the docker daemon as $(whoami)." >&2
    echo "  Add this user to the docker group or grant passwordless sudo for docker," >&2
    echo "  then re-run, or force it with DOCKER=\"sudo docker\"." >&2
    exit 1
  fi
fi

stop() { echo "stopping ${CONTAINER}"; $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true; }

tail_log() {
  local n=${1:-60}
  [ -f "$LOG" ] || { echo "  (no log yet at $LOG)"; return; }
  tail -n "$n" "$LOG" 2>/dev/null || sudo -n tail -n "$n" "$LOG" 2>/dev/null || echo "  ($LOG not readable)"
}

if [ "${1:-}" = "stop" ]; then stop; exit 0; fi

mkdir -p "$LOG_DIR"
$DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true

# GPU access is the explicit non-privileged ROCm set: the kfd/dri device nodes plus
# the video/render groups, no --privileged. No --cap-add SYS_PTRACE either; that one
# is only for running rocgdb/strace inside the container, the server doesn't need it.
cid=$($DOCKER run -d --name "$CONTAINER" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size 64g \
  -v "$MODEL_PATH":"$MODEL_PATH" \
  "$IMAGE" sleep infinity)
if [ "$?" != "0" ] || [ -z "$cid" ]; then
  echo "ERROR: 'docker run' failed via '$DOCKER'. Nothing is serving." >&2
  exit 1
fi

$DOCKER exec -d "$CONTAINER" bash -lc "
cd /
CUDA_VISIBLE_DEVICES=$GPUS \
SGLANG_OPT_USE_TOPK_V2=0 \
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --served-model-name $SERVED_NAME \
  --host $HOST --port $PORT \
  --tp-size $TP_SIZE \
  --mem-fraction-static $MEM_FRACTION \
  --trust-remote-code \
  --watchdog-timeout 1200 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --log-level info \
  > $LOG 2>&1
"
if [ "$?" != "0" ]; then
  echo "ERROR: failed to launch the server in the container." >&2
  $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit 1
fi

echo "container: $CONTAINER"
echo "image:     $IMAGE"
echo "model:     $MODEL_PATH"
echo "gpus:      CUDA_VISIBLE_DEVICES=$GPUS  tp=$TP_SIZE  mem-fraction=$MEM_FRACTION"
echo "log:       $LOG"
echo "base url:  $BASE_URL"
echo "waiting for /health (up to ${READY_TIMEOUT}s; first launch pays weight-load + JIT)..."

start=$(date +%s); ready=0
while [ $(( $(date +%s) - start )) -lt "$READY_TIMEOUT" ]; do
  if $DOCKER exec "$CONTAINER" bash -lc "curl -sf ${BASE_URL}/health >/dev/null 2>&1"; then ready=1; break; fi
  if ! $DOCKER ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: container ${CONTAINER} exited. tail of ${LOG}:"; tail_log 40; exit 1
  fi
  if [ $(( $(date +%s) - start )) -gt 30 ] \
     && ! $DOCKER exec "$CONTAINER" bash -lc "pgrep -f sglang.launch_server >/dev/null 2>&1"; then
    echo "ERROR: the sglang server process exited before /health came up. tail of ${LOG}:"; tail_log 60; exit 1
  fi
  sleep 5
done

if [ "$ready" != "1" ]; then
  echo "ERROR: server did not become ready within ${READY_TIMEOUT}s. tail of ${LOG}:"; tail_log 60; exit 1
fi

echo "READY after $(( $(date +%s) - start ))s"
# absorb the one-time MoE + sampling JIT so later requests are warm
$DOCKER exec "$CONTAINER" bash -lc "curl -s ${BASE_URL}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0}'" >/dev/null 2>&1 || true

echo ""
echo "GLM-5.2-FP8 is up at ${BASE_URL}  (models: ${BASE_URL}/v1/models)"
echo "stop with: $0 stop"
