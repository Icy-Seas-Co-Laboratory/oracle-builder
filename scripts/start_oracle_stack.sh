#!/usr/bin/env bash
# Start the local Oracle Builder compute, orchestration, and web UI stack.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ORACLE_RUNTIME_DIR:-$ROOT_DIR/.oracle-runtime}"
HOST="${ORACLE_HOST:-127.0.0.1}"
SERVE_PORT="${ORACLE_SERVE_PORT:-8100}"
ORCHESTRATOR_PORT="${ORACLE_ORCHESTRATOR_PORT:-8110}"
WEBGUI_PORT="${ORACLE_WEBGUI_PORT:-5111}"
SERVE_URL="http://${HOST}:${SERVE_PORT}"
ORCHESTRATOR_URL="http://${HOST}:${ORCHESTRATOR_PORT}"
LOG_DIR="$RUNTIME_DIR/logs"
SERVE_PID=""
ORCHESTRATOR_PID=""
WEBGUI_PID=""

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command is unavailable: $1" >&2; exit 1; }
}

require_free_port() {
  local port="$1"
  python3 - "$HOST" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"Port {port} on {host} is unavailable: {exc}")
PY
}

wait_for() {
  local name="$1" url="$2" pid="$3"
  local attempt
  for attempt in $(seq 1 60); do
    if curl --fail --silent --show-error "$url" >/dev/null 2>&1; then
      echo "$name is ready at ${url%/health/*}"
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name stopped before becoming ready. See $LOG_DIR/${name}.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "$name did not become ready within 60 seconds. See $LOG_DIR/${name}.log" >&2
  return 1
}

is_online() {
  curl --fail --silent --show-error "$1" >/dev/null 2>&1
}

stop_process() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_process "$WEBGUI_PID"
  stop_process "$ORCHESTRATOR_PID"
  stop_process "$SERVE_PID"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_command uv
require_command npm
require_command curl
require_command python3
mkdir -p "$LOG_DIR" "$RUNTIME_DIR/artifacts"

# Reuse healthy development services. A port occupied by anything else remains
# an error: it is unsafe to assume that an arbitrary process is Oracle Builder.
SERVE_RUNNING=0
ORCHESTRATOR_RUNNING=0
if is_online "$SERVE_URL/health/ready"; then
  SERVE_RUNNING=1
  echo "Reusing oracle-serve at $SERVE_URL"
else
  require_free_port "$SERVE_PORT"
fi
if is_online "$ORCHESTRATOR_URL/health/live"; then
  # A live endpoint alone is not enough: the GUI relies on the model setup
  # schema for architecture defaults and previews. Do not silently reuse an
  # older process that predates that API.
  if is_online "$ORCHESTRATOR_URL/v1/model-setups/simple_cnn"; then
    ORCHESTRATOR_RUNNING=1
    echo "Reusing oracle-orchestrator at $ORCHESTRATOR_URL"
  else
    echo "An older oracle-orchestrator is running at $ORCHESTRATOR_URL." >&2
    echo "Restart it from this checkout so the model setup API is available, then run this script again." >&2
    exit 1
  fi
else
  require_free_port "$ORCHESTRATOR_PORT"
fi
require_free_port "$WEBGUI_PORT"

if [[ "${ORACLE_STACK_SKIP_SETUP:-0}" != "1" ]]; then
  echo "Synchronizing Python API dependencies…"
  (cd "$ROOT_DIR" && uv sync --extra api --locked)
  echo "Synchronizing web GUI dependencies…"
  (cd "$ROOT_DIR/webgui" && npm ci)
fi

if [[ "$SERVE_RUNNING" == "0" ]]; then
  echo "Starting oracle-serve…"
  (
    cd "$ROOT_DIR"
    exec uv run --extra api oracle-serve --host "$HOST" --port "$SERVE_PORT" --worker-id local
  ) >"$LOG_DIR/oracle-serve.log" 2>&1 &
  SERVE_PID=$!
  wait_for "oracle-serve" "$SERVE_URL/health/ready" "$SERVE_PID"
fi

if [[ "$ORCHESTRATOR_RUNNING" == "0" ]]; then
  echo "Starting oracle-orchestrator…"
  (
    cd "$ROOT_DIR"
    exec uv run --extra api oracle-orchestrator \
      --database "$RUNTIME_DIR/orchestrator.sqlite" \
      --workspace-root "$ROOT_DIR" \
      --artifact-root "$RUNTIME_DIR/artifacts" \
      --oracle-serve "Local=${SERVE_URL}" \
      --host "$HOST" --port "$ORCHESTRATOR_PORT"
  ) >"$LOG_DIR/orchestrator.log" 2>&1 &
  ORCHESTRATOR_PID=$!
  wait_for "orchestrator" "$ORCHESTRATOR_URL/health/live" "$ORCHESTRATOR_PID"
fi

echo "Starting web GUI…"
(
  cd "$ROOT_DIR/webgui"
  export ORCHESTRATOR_URL
  exec npm run dev -- --host "$HOST" --port "$WEBGUI_PORT"
) >"$LOG_DIR/webgui.log" 2>&1 &
WEBGUI_PID=$!
wait_for "webgui" "http://${HOST}:${WEBGUI_PORT}/" "$WEBGUI_PID"

cat <<EOF

Oracle Builder stack is running.
  Web GUI:       http://${HOST}:${WEBGUI_PORT}/
  Orchestrator:  ${ORCHESTRATOR_URL}
  Compute API:   ${SERVE_URL}
  Runtime data:  ${RUNTIME_DIR}

Press Ctrl-C to stop the stack. Logs are in ${LOG_DIR}.
EOF

wait "$WEBGUI_PID"
