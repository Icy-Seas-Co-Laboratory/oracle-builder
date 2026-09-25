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
ACCELERATOR_REQUEST="${ORACLE_ACCELERATOR:-auto}"
ACCELERATOR="cpu"
GPU_EXTRA=""
KERNEL_NAME="$(uname -s)"
NVIDIA_SMI="${ORACLE_NVIDIA_SMI:-nvidia-smi}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command is unavailable: $1" >&2; exit 1; }
}

is_wsl() {
  [[ "$KERNEL_NAME" == "Linux" ]] && {
    [[ "$(uname -r)" == *[Mm]icrosoft* ]] || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null
  }
}

has_nvidia_gpu() {
  command -v "$NVIDIA_SMI" >/dev/null 2>&1 && "$NVIDIA_SMI" --query-gpu=index --format=csv,noheader >/dev/null 2>&1
}

configure_accelerator() {
  case "$ACCELERATOR_REQUEST" in
    auto|cpu|cuda|metal) ;;
    *) echo "ORACLE_ACCELERATOR must be auto, cpu, cuda, or metal (got $ACCELERATOR_REQUEST)" >&2; exit 2 ;;
  esac
  case "$KERNEL_NAME" in
    MINGW*|MSYS*|CYGWIN*)
      echo "Native Windows shells are not supported for GPU training. Run this script from WSL2 instead." >&2
      exit 2
      ;;
  esac

  if [[ "$ACCELERATOR_REQUEST" == "cpu" ]]; then
    ACCELERATOR="cpu"
  elif [[ "$ACCELERATOR_REQUEST" == "metal" ]]; then
    if [[ "$KERNEL_NAME" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
      echo "Metal acceleration requires Apple Silicon macOS." >&2
      exit 2
    fi
    ACCELERATOR="metal"
  elif [[ "$ACCELERATOR_REQUEST" == "cuda" ]]; then
    if [[ "$KERNEL_NAME" != "Linux" ]]; then
      echo "CUDA acceleration is supported by this launcher on Linux or WSL2." >&2
      exit 2
    fi
    if ! has_nvidia_gpu; then
      echo "CUDA was requested but nvidia-smi cannot query a usable GPU. Check the NVIDIA driver/WSL GPU passthrough." >&2
      exit 2
    fi
    ACCELERATOR="cuda"
  elif [[ "$KERNEL_NAME" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    ACCELERATOR="metal"
  elif [[ "$KERNEL_NAME" == "Linux" ]] && has_nvidia_gpu; then
    ACCELERATOR="cuda"
  fi

  case "$ACCELERATOR" in
    cuda)
      GPU_EXTRA="gpu-wsl2"
      if ! is_wsl; then GPU_EXTRA="gpu-linux"; fi
      # Compute inventory intentionally honors CUDA_VISIBLE_DEVICES. Populate
      # it from the host only when the user has not already constrained it.
      if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        CUDA_VISIBLE_DEVICES="$($NVIDIA_SMI --query-gpu=index --format=csv,noheader | paste -sd, -)"
        export CUDA_VISIBLE_DEVICES
      fi
      export ORACLE_ACCELERATOR_BACKEND="cuda"
      ;;
    metal)
      GPU_EXTRA="gpu-macos"
      export ORACLE_ACCELERATOR_BACKEND="metal"
      ;;
    cpu)
      export ORACLE_ACCELERATOR_BACKEND="cpu"
      # Make an explicit CPU request real on CUDA hosts.  Metal is selected by
      # TensorFlow rather than this variable; queue-level CPU allocation still
      # uses the runtime's CPU distribution strategy on macOS.
      if [[ "$KERNEL_NAME" == "Linux" ]]; then export CUDA_VISIBLE_DEVICES="-1"; fi
      ;;
  esac
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

has_startup_reconciliation() {
  curl --fail --silent --show-error "$ORCHESTRATOR_URL/health/ready" | python3 -c '
import json
import sys
try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    raise SystemExit(1)
raise SystemExit(0 if "startup_reconciliation" in payload else 1)
'
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
configure_accelerator
mkdir -p "$LOG_DIR" "$RUNTIME_DIR/artifacts" "$ROOT_DIR/runs" "$ROOT_DIR/datasets"

echo "Accelerator: $ACCELERATOR${GPU_EXTRA:+ (uv extra: $GPU_EXTRA)}"
if [[ "$ACCELERATOR" == "cuda" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
elif [[ "$ACCELERATOR_REQUEST" == "auto" && "$ACCELERATOR" == "cpu" ]]; then
  echo "No supported accelerator detected; using CPU. Set ORACLE_ACCELERATOR=cpu to make CPU selection explicit."
fi

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
  if is_online "$ORCHESTRATOR_URL/v1/model-setups/simple_cnn" && has_startup_reconciliation; then
    ORCHESTRATOR_RUNNING=1
    echo "Reusing oracle-orchestrator at $ORCHESTRATOR_URL"
  else
    echo "An older oracle-orchestrator is running at $ORCHESTRATOR_URL." >&2
    echo "Restart it from this checkout so model setup and startup reconciliation are available, then run this script again." >&2
    exit 1
  fi
else
  require_free_port "$ORCHESTRATOR_PORT"
fi
require_free_port "$WEBGUI_PORT"

if [[ "${ORACLE_STACK_SKIP_SETUP:-0}" != "1" ]]; then
  echo "Synchronizing Python API dependencies…"
  UV_SYNC_ARGS=(--extra api --locked)
  if [[ -n "$GPU_EXTRA" ]]; then UV_SYNC_ARGS+=(--extra "$GPU_EXTRA"); fi
  (cd "$ROOT_DIR" && uv sync "${UV_SYNC_ARGS[@]}")
  echo "Synchronizing web GUI dependencies…"
  (cd "$ROOT_DIR/webgui" && npm ci)
fi

if [[ "${ORACLE_STACK_SKIP_ACCELERATOR_CHECK:-0}" != "1" ]]; then
  echo "Checking TensorFlow accelerator visibility…"
  if ! (cd "$ROOT_DIR" && uv run python scripts/check_tensorflow_devices.py); then
    echo "Accelerator verification failed; the stack will continue, but compute may fall back to CPU. See docs/operations-and-troubleshooting.md." >&2
  fi
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
      --runs-root "$ROOT_DIR/runs" \
      --datasets-root "$ROOT_DIR/datasets" \
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
  Accelerator:   ${ACCELERATOR}

Press Ctrl-C to stop the stack. Logs are in ${LOG_DIR}.
EOF

wait "$WEBGUI_PID"
