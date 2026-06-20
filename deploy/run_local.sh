#!/usr/bin/env bash
# Veritas local federation launcher.
#
# Starts the Tier-2 control plane (:9000) plus N bank nodes (:8100+i) as local
# background processes with the correct env, waits for health, prints URLs.
#
#   deploy/run_local.sh up [N]      # start plane + N nodes (default 4)
#   deploy/run_local.sh down        # stop everything started by `up`
#   deploy/run_local.sh status      # show health of plane + nodes
#
# Env knobs (with defaults):
#   VERITAS_ADMIN_KEY=dev-admin-key       admin secret (X-Admin-Key)
#   VERITAS_MIN_UPDATES=3                 plane auto-aggregation threshold
#   VERITAS_AUTOSTART_FEDERATION=1        nodes run their own federation loop
#   VERITAS_POLL_INTERVAL=2               node loop cadence (seconds)
#
# The e2e harness (deploy/e2e.py) drives aggregation explicitly via
# POST /v1/rounds/advance, so it launches the stack with
# VERITAS_AUTOSTART_FEDERATION=0 VERITAS_MIN_UPDATES=99.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CP_DIR="$ROOT/controlplane"
NODE_DIR="$ROOT/node"
RUN_DIR="${VERITAS_RUN_DIR:-/tmp/veritas-deploy}"
PID_DIR="$RUN_DIR/pids"
LOG_DIR="$RUN_DIR/logs"

ADMIN_KEY="${VERITAS_ADMIN_KEY:-dev-admin-key}"
MIN_UPDATES="${VERITAS_MIN_UPDATES:-3}"
AUTOSTART="${VERITAS_AUTOSTART_FEDERATION:-1}"
POLL_INTERVAL="${VERITAS_POLL_INTERVAL:-2}"
# Demo operating point: pin a utility-preserving noise multiplier on the tiny
# 11-dim reference model. The DP accounting is still rigorous — /v1/privacy
# reports the TRUE ε this σ yields. Strict-privacy default (calibrate σ from
# ε=8) is correct for production-scale models but destroys this toy model, the
# real privacy/utility tradeoff. Override with VERITAS_DP_SIGMA=... for a demo,
# or unset + set VERITAS_DP_EPSILON to exercise full ε-calibration.
DP_SIGMA="${VERITAS_DP_SIGMA:-0.1}"
PLANE_PORT=9000
PLANE_URL="http://localhost:${PLANE_PORT}"

mkdir -p "$PID_DIR" "$LOG_DIR"

_wait_health() {  # url, name
  local url="$1" name="$2" i
  for i in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "ERROR: $name did not become healthy at $url" >&2
  return 1
}

up() {
  local n="${1:-4}"
  echo "Starting Veritas control plane + $n node(s) ..."

  # --- control plane ---
  (
    cd "$CP_DIR"
    # shellcheck disable=SC1091
    . .venv/bin/activate
    VERITAS_ADMIN_KEY="$ADMIN_KEY" VERITAS_MIN_UPDATES="$MIN_UPDATES" \
      VERITAS_DP_SIGMA="$DP_SIGMA" \
      nohup uvicorn controlplane.server.app:app --port "$PLANE_PORT" \
      > "$LOG_DIR/controlplane.log" 2>&1 &
    echo $! > "$PID_DIR/controlplane.pid"
  )
  _wait_health "$PLANE_URL/health" "control plane"
  echo "  control plane  -> $PLANE_URL  (pid $(cat "$PID_DIR/controlplane.pid"))"

  # --- nodes ---
  # Optional: VERITAS_BLIND_NODE=<i> makes node i train on the campaign-FREE
  # synthetic fallback (by pointing its feature map at a non-existent file) so
  # it is a genuinely "non-seeing" bank — targeted by a cross-institution
  # campaign in its eval, but blind in its siloed model. The federated model
  # must carry the campaign signal in from the seeing banks. Used by e2e.py to
  # prove the federated-vs-siloed lift; harmless when unset.
  local blind="${VERITAS_BLIND_NODE:-}"
  local i port fmap_env
  for i in $(seq 0 $((n - 1))); do
    port=$((8100 + i))
    fmap_env=""
    if [ -n "$blind" ] && [ "$i" = "$blind" ]; then
      fmap_env="VERITAS_FEATURE_MAP=/nonexistent/veritas-blind-node.yaml"
    fi
    (
      cd "$NODE_DIR"
      # shellcheck disable=SC1091
      . .venv/bin/activate
      env $fmap_env \
      VERITAS_NODE_ID="node$i" VERITAS_NODE_INDEX="$i" VERITAS_PORT="$port" \
        VERITAS_PLANE_URL="$PLANE_URL" VERITAS_ADMIN_KEY="$ADMIN_KEY" \
        VERITAS_AUTOSTART_FEDERATION="$AUTOSTART" VERITAS_POLL_INTERVAL="$POLL_INTERVAL" \
        nohup uvicorn node.server.app:app --port "$port" \
        > "$LOG_DIR/node$i.log" 2>&1 &
      echo $! > "$PID_DIR/node$i.pid"
    )
    _wait_health "http://localhost:$port/health" "node$i"
    echo "  node$i          -> http://localhost:$port  (pid $(cat "$PID_DIR/node$i.pid"))"
  done

  echo
  echo "Stack up. Members enrol as 'pending' — approve them to start federating:"
  echo "  for m in $(seq -f 'node%g' 0 $((n - 1))); do \\"
  echo "    curl -s -X POST -H \"X-Admin-Key: $ADMIN_KEY\" $PLANE_URL/v1/members/\$m/approve; done"
  echo
  echo "URLs: plane $PLANE_URL  |  nodes http://localhost:8100..$((8100 + n - 1))"
  echo "Logs: $LOG_DIR   |   Teardown: deploy/run_local.sh down"
}

down() {
  echo "Stopping Veritas stack ..."
  shopt -s nullglob
  for f in "$PID_DIR"/*.pid; do
    pid="$(cat "$f")"
    if kill "$pid" >/dev/null 2>&1; then
      echo "  killed $(basename "$f" .pid) (pid $pid)"
    fi
    rm -f "$f"
  done
  # Belt-and-braces: free the well-known ports if anything lingers.
  for port in 9000 8100 8101 8102 8103 8104 8105; do
    pid="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    [ -n "$pid" ] && kill "$pid" >/dev/null 2>&1 && echo "  freed port $port (pid $pid)" || true
  done
  echo "Done."
}

status() {
  echo "control plane: $(curl -fsS "$PLANE_URL/health" 2>/dev/null || echo DOWN)"
  for i in 0 1 2 3 4 5; do
    port=$((8100 + i))
    h="$(curl -fsS "http://localhost:$port/health" 2>/dev/null || true)"
    [ -n "$h" ] && echo "node$i ($port): $h"
  done
}

cmd="${1:-up}"; shift || true
case "$cmd" in
  up)     up "$@" ;;
  down)   down ;;
  status) status ;;
  *) echo "usage: $0 {up [N]|down|status}" >&2; exit 2 ;;
esac
