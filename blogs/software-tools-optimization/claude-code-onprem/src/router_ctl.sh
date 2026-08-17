#!/usr/bin/env bash
# Manage the LiteLLM GLM-5.2 router on the node (127.0.0.1:PORT -> SGLang :31090).
# Self-locating: BASE is the directory this script lives in, so it works wherever
# you drop it (your own home, /data, /tmp, ...). No hardcoded user or path.
# Port, key, config, and venv come from router.env next to this script (written
# by glm-setup) or from the environment, with sane defaults.
# Author: adlashab <adlashab@amd.com>
set -u

BASE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$BASE/router.env" ] && . "$BASE/router.env"

CFG="${GLM_ROUTER_CONFIG:-$BASE/litellm/config.yaml}"
VENV="${GLM_ROUTER_VENV:-$BASE/litellm/venv}"
LOG="${GLM_ROUTER_LOG:-$BASE/logs/litellm.log}"
PORT="${GLM_ROUTER_PORT:-4000}"
KEY="${GLM_ROUTER_KEY:-sk-glm-local}"

up(){ ss -ltn 2>/dev/null | grep -q ":$PORT "; }

case "${1:-status}" in
  start)
    if up; then echo "router already up on :$PORT"; exit 0; fi
    if [ ! -f "$CFG" ]; then echo "config not found: $CFG" >&2; exit 1; fi
    if [ ! -x "$VENV/bin/litellm" ] && [ ! -f "$VENV/bin/activate" ]; then
      echo "venv not found at $VENV (run glm-setup, or create it and pip install 'litellm[proxy]')" >&2; exit 1
    fi
    mkdir -p "$(dirname "$LOG")"
    setsid bash -c "source '$VENV/bin/activate'; exec litellm --config '$CFG' --host 127.0.0.1 --port $PORT" </dev/null >"$LOG" 2>&1 &
    for i in $(seq 1 60); do up && break; sleep 1; done
    up && echo "router started on :$PORT (pid $(pgrep -f "litellm --config $CFG" | head -1))" \
       || { echo "FAILED to start; tail log:"; tail -n 20 "$LOG"; exit 1; }
    ;;
  stop)
    pkill -f "litellm --config $CFG" && echo "router stopped" || echo "router not running"
    ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status)
    if up; then
      echo "router UP on :$PORT (pid $(pgrep -f "litellm --config $CFG" | head -1))"
      curl -s -m 10 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models"
      echo
    else echo "router DOWN on :$PORT"; fi
    ;;
  *) echo "usage: $0 {start|stop|restart|status}";;
esac
