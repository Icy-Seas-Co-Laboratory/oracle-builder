#!/usr/bin/env bash
# Start the local Oracle Builder pull-worker, orchestration, and web UI stack.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ORACLE_RUNTIME_DIR:-$ROOT_DIR/.oracle-runtime}"
HOST="${ORACLE_HOST:-127.0.0.1}"
ORCHESTRATOR_PORT="${ORACLE_ORCHESTRATOR_PORT:-8110}"
INFERENCE_PORT="${ORACLE_INFERENCE_PORT:-8111}"
WEBGUI_PORT="${ORACLE_WEBGUI_PORT:-5111}"
STARTUP_TIMEOUT_SECONDS="${ORACLE_STARTUP_TIMEOUT_SECONDS:-300}"
ORCHESTRATOR_URL="http://${HOST}:${ORCHESTRATOR_PORT}"
LOG_DIR="$RUNTIME_DIR/logs"
WORKER_PIDS=()
ORCHESTRATOR_PID=""
INFERENCE_PID=""
WEBGUI_PID=""
LOCAL_JOIN_TOKEN_FILE=""
ACCELERATOR_REQUEST="${ORACLE_ACCELERATOR:-auto}"
ACCELERATOR="cpu"
GPU_EXTRA=""
KERNEL_NAME="$(uname -s)"
NVIDIA_SMI="${ORACLE_NVIDIA_SMI:-nvidia-smi}"
# The unprefixed names are the launcher contract.  Keep the original Builder
# names as a compatibility fallback for deployments that set them directly.
LOCAL_WORKER_SLOTS="${ORACLE_LOCAL_WORKER_SLOTS:-1}"
LOCAL_WORKER_CPU_CAPACITY="${ORACLE_LOCAL_WORKER_CPU_CAPACITY:-}"
WORKER_ARTIFACT_TIMEOUT_SECONDS="${ORACLE_WORKER_ARTIFACT_TIMEOUT_SECONDS:-900}"
ARTIFACT_S3_BUCKET="${ORACLE_ARTIFACT_S3_BUCKET:-}"
ARTIFACT_S3_PREFIX="${ORACLE_ARTIFACT_S3_PREFIX:-oracle-builder}"
ARTIFACT_S3_ENDPOINT_URL="${ORACLE_ARTIFACT_S3_ENDPOINT_URL:-}"
ARTIFACT_S3_REGION="${ORACLE_ARTIFACT_S3_REGION:-}"
WORKER_DEPLOYMENT_PROFILES="${ORACLE_WORKER_DEPLOYMENT_PROFILES:-}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command is unavailable: $1" >&2; exit 1; }
}

require_positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer (got $value)" >&2
    exit 2
  }
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
  for attempt in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
    if curl --noproxy '*' --fail --silent --show-error "$url" >/dev/null 2>&1; then
      echo "$name is ready at ${url%/health/*}"
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name stopped before becoming ready. See $LOG_DIR/${name}.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "$name did not become ready within ${STARTUP_TIMEOUT_SECONDS} seconds. See $LOG_DIR/${name}.log" >&2
  return 1
}

is_online() {
  curl --noproxy '*' --fail --silent --show-error "$1" >/dev/null 2>&1
}

has_startup_reconciliation() {
  curl --noproxy '*' --fail --silent --show-error "$ORCHESTRATOR_URL/health/ready" | python3 -c '
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
  stop_process "$INFERENCE_PID"
  # Bash 3.2 (the system Bash on macOS) treats expansion of an empty indexed
  # array as an unset variable under `set -u`. The startup can fail before any
  # workers are launched, so guard the expansion used by the cleanup trap.
  if [[ -n "${WORKER_PIDS[*]-}" ]]; then
    for worker_pid in "${WORKER_PIDS[@]}"; do stop_process "$worker_pid"; done
  fi
  if [[ -n "$LOCAL_JOIN_TOKEN_FILE" && -f "$LOCAL_JOIN_TOKEN_FILE" ]]; then rm -f "$LOCAL_JOIN_TOKEN_FILE"; fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_command uv
require_command npm
require_command curl
require_command python3
require_positive_integer "ORACLE_LOCAL_WORKER_SLOTS" "$LOCAL_WORKER_SLOTS"
require_positive_integer "ORACLE_STARTUP_TIMEOUT_SECONDS" "$STARTUP_TIMEOUT_SECONDS"
require_positive_integer "ORACLE_INFERENCE_PORT" "$INFERENCE_PORT"
require_positive_integer "ORACLE_WORKER_ARTIFACT_TIMEOUT_SECONDS" "$WORKER_ARTIFACT_TIMEOUT_SECONDS"
if [[ -n "$LOCAL_WORKER_CPU_CAPACITY" ]]; then
  require_positive_integer "ORACLE_LOCAL_WORKER_CPU_CAPACITY" "$LOCAL_WORKER_CPU_CAPACITY"
fi
configure_accelerator
mkdir -p "$LOG_DIR" "$RUNTIME_DIR/artifacts" "$ROOT_DIR/runs" "$ROOT_DIR/datasets"

echo "Accelerator: $ACCELERATOR${GPU_EXTRA:+ (uv extra: $GPU_EXTRA)}"
echo "Local worker capacity: ${LOCAL_WORKER_SLOTS} process slot(s)${LOCAL_WORKER_CPU_CAPACITY:+, ${LOCAL_WORKER_CPU_CAPACITY} CPU core(s)}"
echo "Worker artifact-transfer timeout: ${WORKER_ARTIFACT_TIMEOUT_SECONDS}s"
if [[ "$ACCELERATOR" == "cuda" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
elif [[ "$ACCELERATOR_REQUEST" == "auto" && "$ACCELERATOR" == "cpu" ]]; then
  echo "No supported accelerator detected; using CPU. Set ORACLE_ACCELERATOR=cpu to make CPU selection explicit."
fi

# Reuse healthy development services. A port occupied by anything else remains
# an error: it is unsafe to assume that an arbitrary process is Oracle Builder.
ORCHESTRATOR_RUNNING=0
if is_online "$ORCHESTRATOR_URL/health/live"; then
  if has_startup_reconciliation; then
    echo "An oracle-orchestrator is already running at $ORCHESTRATOR_URL." >&2
    echo "Stop that stack before starting this version so the V2 runtime token and proxy can be configured together." >&2
    exit 1
  else
    echo "An older oracle-orchestrator is running at $ORCHESTRATOR_URL." >&2
    echo "Restart it from this checkout so model setup and startup reconciliation are available, then run this script again." >&2
    exit 1
  fi
else
  require_free_port "$ORCHESTRATOR_PORT"
fi
require_free_port "$WEBGUI_PORT"
require_free_port "$INFERENCE_PORT"

# A process-local secret protects the loopback inference service from callers
# that bypass the orchestrator's catalog and role checks.
INFERENCE_INTERNAL_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

if [[ "${ORACLE_STACK_SKIP_SETUP:-0}" != "1" ]]; then
  echo "Synchronizing Python API dependencies…"
  UV_SYNC_ARGS=(--extra api --locked)
  if [[ -n "$ARTIFACT_S3_BUCKET" ]]; then UV_SYNC_ARGS+=(--extra storage-s3); fi
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

if [[ "$ORCHESTRATOR_RUNNING" == "0" ]]; then
  echo "Starting oracle-orchestrator…"
  # Keep one non-empty argv array. Bash 3 with `set -u` treats an empty
  # indexed array expansion as unset, which broke the no-S3/no-profile local
  # startup path on macOS.
  ORCHESTRATOR_ARGS=(
    --database "$RUNTIME_DIR/orchestrator.sqlite"
    --workspace-root "$ROOT_DIR"
    --artifact-root "$RUNTIME_DIR/artifacts"
    --log-root "$LOG_DIR"
    --runs-root "$ROOT_DIR/runs"
    --datasets-root "$ROOT_DIR/datasets"
    --host "$HOST"
    --port "$ORCHESTRATOR_PORT"
  )
  if [[ -n "$ARTIFACT_S3_BUCKET" ]]; then
    ORCHESTRATOR_ARGS+=(--s3-bucket "$ARTIFACT_S3_BUCKET" --s3-prefix "$ARTIFACT_S3_PREFIX")
    if [[ -n "$ARTIFACT_S3_ENDPOINT_URL" ]]; then ORCHESTRATOR_ARGS+=(--s3-endpoint-url "$ARTIFACT_S3_ENDPOINT_URL"); fi
    if [[ -n "$ARTIFACT_S3_REGION" ]]; then ORCHESTRATOR_ARGS+=(--s3-region "$ARTIFACT_S3_REGION"); fi
  fi
  if [[ -n "$WORKER_DEPLOYMENT_PROFILES" ]]; then
    ORCHESTRATOR_ARGS+=(--worker-deployment-profiles "$WORKER_DEPLOYMENT_PROFILES")
  fi
  if [[ -n "${ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256:-}" ]]; then
    ORCHESTRATOR_ARGS+=(--role-tokens-sha256 "$ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256")
  elif [[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" || "$HOST" == "::1" ]]; then
    # This launcher is a single-user local-development convenience. Remote
    # deployments must provide hashed role tokens instead of this bypass.
    ORCHESTRATOR_ARGS+=(--allow-unauthenticated-mutations)
  else
    echo "ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256 is required when binding the stack remotely." >&2
    exit 2
  fi
  (
    cd "$ROOT_DIR"
    export ORACLE_INFERENCE_RUNTIME_URL="http://127.0.0.1:${INFERENCE_PORT}"
    export ORACLE_INFERENCE_RUNTIME_TOKEN="$INFERENCE_INTERNAL_TOKEN"
    ORCHESTRATOR_EXTRAS=(--extra api)
    if [[ -n "$ARTIFACT_S3_BUCKET" ]]; then ORCHESTRATOR_EXTRAS+=(--extra storage-s3); fi
    exec uv run "${ORCHESTRATOR_EXTRAS[@]}" oracle-orchestrator "${ORCHESTRATOR_ARGS[@]}"
  ) >"$LOG_DIR/orchestrator.log" 2>&1 &
  ORCHESTRATOR_PID=$!
  wait_for "orchestrator" "$ORCHESTRATOR_URL/health/live" "$ORCHESTRATOR_PID"
fi

echo "Starting resident inference runtime…"
(
  cd "$ROOT_DIR"
  export ORACLE_INFERENCE_RUNTIME_TOKEN="$INFERENCE_INTERNAL_TOKEN"
  if [[ -n "${ORACLE_INFERENCE_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$ORACLE_INFERENCE_CUDA_VISIBLE_DEVICES"
  fi
  exec uv run --extra api oracle-inference-runtime \
    --database "$RUNTIME_DIR/orchestrator.sqlite" \
    --workspace-root "$ROOT_DIR" \
    --artifact-root "$RUNTIME_DIR/artifacts" \
    --host 127.0.0.1 \
    --port "$INFERENCE_PORT"
) >"$LOG_DIR/inference-runtime.log" 2>&1 &
INFERENCE_PID=$!
wait_for "inference-runtime" "http://127.0.0.1:${INFERENCE_PORT}/internal/v2/health" "$INFERENCE_PID"

# Keep the worker in its existing pool across restarts. Queued work is scoped
# to a pool, and the private worker credentials are sufficient to rejoin it.
LOCAL_POOL_NAME="local-$(date +%s)-$$"
ORCHESTRATOR_CURL_ARGS=(-H 'Content-Type: application/json')
if [[ -n "${ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256:-}" ]]; then
  if [[ -z "${ORACLE_ORCHESTRATOR_TOKEN:-}" ]]; then
    echo "ORACLE_ORCHESTRATOR_TOKEN is required when role-token security is configured." >&2
    exit 2
  fi
  ORCHESTRATOR_CURL_ARGS+=(-H "Authorization: Bearer ${ORACLE_ORCHESTRATOR_TOKEN}")
fi
LOCAL_POOL_IDS=()
LOCAL_POOL_IDS_OUTPUT="$(ORACLE_LOCAL_POOL_URL="$ORCHESTRATOR_URL" ORACLE_LOCAL_POOL_TOKEN="${ORACLE_ORCHESTRATOR_TOKEN:-}" python3 - "$RUNTIME_DIR/workers" <<'PY'
import json, os, pathlib, sys, urllib.request

root = pathlib.Path(sys.argv[1])
base = os.environ['ORACLE_LOCAL_POOL_URL'].rstrip('/')
headers = {}
if os.environ.get('ORACLE_LOCAL_POOL_TOKEN'):
    headers['Authorization'] = 'Bearer ' + os.environ['ORACLE_LOCAL_POOL_TOKEN']

def fetch(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=5) as response:
        return json.load(response)

try:
    pools = {item['pool_id']: item for item in fetch('/v1/worker-pools')['pools']}
    jobs = fetch('/v1/jobs')['jobs']
except (OSError, ValueError, KeyError):
    sys.exit(0)

pending = {}
for job in jobs:
    if job.get('status') in {'queued', 'leased', 'running', 'paused', 'cancel_requested'} and job.get('worker_pool_id'):
        pool_id = job['worker_pool_id']
        pending[pool_id] = pending.get(pool_id, 0) + 1
try:
    queued_runs = fetch('/v1/queued-runs')['queued_runs']
except (OSError, ValueError, KeyError):
    queued_runs = []
for run in queued_runs:
    if run.get('status') in {'pending_verification', 'verifying', 'ready', 'queued', 'running', 'paused', 'waiting_for_resources'} and run.get('worker_pool_id'):
        pool_id = run['worker_pool_id']
        pending[pool_id] = pending.get(pool_id, 0) + 1

candidates = []
for path in root.glob('*.json'):
    try:
        if path.is_symlink() or path.stat().st_mode & 0o077:
            continue
        credential = json.loads(path.read_text())
        pool_id = credential['pool_id']
        pool = pools.get(pool_id)
        if (credential['orchestrator_url'].rstrip('/') == base and pool
                and pool.get('enabled') and {'train', 'infer'} <= set(pool.get('allowed_actions') or [])):
            candidates.append((pending.get(pool_id, 0), path.stat().st_mtime, pool_id))
    except (OSError, ValueError, KeyError, TypeError):
        continue
if candidates:
    needed = sorted((candidate for candidate in candidates if candidate[0]), reverse=True)
    for _, _, pool_id in needed or [max(candidates)]:
        print(pool_id)
PY
)"
while IFS= read -r pool_id; do
  [[ -n "$pool_id" ]] && LOCAL_POOL_IDS+=("$pool_id")
done <<< "$LOCAL_POOL_IDS_OUTPUT"
if (( ${#LOCAL_POOL_IDS[@]} )); then
  echo "Reusing local worker pools: ${LOCAL_POOL_IDS[*]}"
else
  POOL_RESPONSE="$(curl --noproxy '*' --fail --silent --show-error \
    "${ORCHESTRATOR_CURL_ARGS[@]}" \
    -d "{\"name\":\"${LOCAL_POOL_NAME}\",\"allowed_actions\":[\"train\",\"infer\"]}" \
    "${ORCHESTRATOR_URL}/v1/worker-pools")"
  read -r LOCAL_POOL_ID LOCAL_JOIN_TOKEN < <(python3 -c '
import json, sys
value = json.load(sys.stdin)
print(value["pool_id"], value["registration_token"])
' <<<"$POOL_RESPONSE")
  LOCAL_POOL_IDS=("$LOCAL_POOL_ID")
fi
# Keep the one-time join secret off the process command line. The worker gets
# it from its environment only for the initial registration, then persists its
# server-issued identity in the owner-only credentials file.
umask 077
mkdir -p "$RUNTIME_DIR/worker-secrets"
if [[ -n "${LOCAL_JOIN_TOKEN:-}" ]]; then
  LOCAL_JOIN_TOKEN_FILE="$RUNTIME_DIR/worker-secrets/${LOCAL_POOL_ID}.join-token"
  printf '%s\n' "$LOCAL_JOIN_TOKEN" >"$LOCAL_JOIN_TOKEN_FILE"
  chmod 600 "$LOCAL_JOIN_TOKEN_FILE"
  unset LOCAL_JOIN_TOKEN
fi

echo "Starting registered local oracle-worker(s)…"
WORKER_CAPABILITIES=(--capability "cpu_capacity=${LOCAL_WORKER_CPU_CAPACITY:-$LOCAL_WORKER_SLOTS}")
if [[ "$ACCELERATOR" == "cuda" ]]; then
  WORKER_CAPABILITIES+=(--capability "gpu_ids=${CUDA_VISIBLE_DEVICES}")
elif [[ "$ACCELERATOR" == "metal" ]]; then
  WORKER_CAPABILITIES+=(--capability "gpu_ids=0")
fi
for LOCAL_POOL_ID in "${LOCAL_POOL_IDS[@]}"; do
(
  cd "$ROOT_DIR"
  export ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN
  if [[ -n "$LOCAL_JOIN_TOKEN_FILE" ]]; then
    ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN="$(<"$LOCAL_JOIN_TOKEN_FILE")"
  else
    unset ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN
  fi
  worker_child=""
  stop_local_worker() {
    trap - TERM INT
    if [[ -n "$worker_child" ]]; then
      kill "$worker_child" 2>/dev/null || true
      wait "$worker_child" 2>/dev/null || true
    fi
    exit 0
  }
  trap stop_local_worker TERM INT
  while true; do
    uv run --extra api oracle-worker \
      --orchestrator-url "$ORCHESTRATOR_URL" \
      --pool-id "$LOCAL_POOL_ID" \
      --name "local-$HOST" \
      --credentials-file "$RUNTIME_DIR/workers/${LOCAL_POOL_ID}.json" \
      --scratch-root "$RUNTIME_DIR/worker-scratch" \
      --artifact-timeout-seconds "$WORKER_ARTIFACT_TIMEOUT_SECONDS" \
      --executor train \
      --executor infer \
      "${WORKER_CAPABILITIES[@]}" &
    worker_child=$!
    if wait "$worker_child"; then
      echo "Local worker exited; restarting in 5 seconds to recover pending work."
    else
      echo "Local worker exited with status $?; restarting in 5 seconds to recover pending work."
    fi
    sleep 5 &
    worker_child=$!
    wait "$worker_child" || true
  done
) >"$LOG_DIR/oracle-worker-${LOCAL_POOL_ID}.log" 2>&1 &
WORKER_PID=$!
WORKER_PIDS+=("$WORKER_PID")
for attempt in $(seq 1 20); do
  if curl --noproxy '*' --fail --silent "$ORCHESTRATOR_URL/v1/workers?pool_id=$LOCAL_POOL_ID" | python3 -c '
import json, sys
from datetime import datetime, timezone
workers = json.load(sys.stdin).get("workers", [])
recent = lambda worker: worker.get("state") != "offline" and (datetime.now(timezone.utc) - datetime.fromisoformat(worker["last_seen_at"])).total_seconds() < 30
raise SystemExit(0 if any(recent(worker) for worker in workers) else 1)
' >/dev/null 2>&1; then
    echo "Local pull worker registered in pool $LOCAL_POOL_ID"
    rm -f "$LOCAL_JOIN_TOKEN_FILE"
    LOCAL_JOIN_TOKEN_FILE=""
    break
  fi
  if ! kill -0 "$WORKER_PID" >/dev/null 2>&1; then
    echo "oracle-worker stopped before registration. See $LOG_DIR/oracle-worker-${LOCAL_POOL_ID}.log" >&2
    exit 1
  fi
  if [[ "$attempt" == "20" ]]; then
    echo "oracle-worker did not register within 20 seconds. See $LOG_DIR/oracle-worker-${LOCAL_POOL_ID}.log" >&2
    exit 1
  fi
  sleep 1
done
done

echo "Starting web GUI…"
(
  cd "$ROOT_DIR/webgui"
  export ORCHESTRATOR_URL
  export ORCHESTRATOR_OPERATOR_TOKEN="${ORACLE_ORCHESTRATOR_TOKEN:-}"
  exec npm run dev -- --host "$HOST" --port "$WEBGUI_PORT"
) >"$LOG_DIR/webgui.log" 2>&1 &
WEBGUI_PID=$!
wait_for "webgui" "http://${HOST}:${WEBGUI_PORT}/" "$WEBGUI_PID"

cat <<EOF

Oracle Builder stack is running.
  Web GUI:       http://${HOST}:${WEBGUI_PORT}/
  Orchestrator:  ${ORCHESTRATOR_URL}
  Local workers: registered pull workers (${LOCAL_POOL_IDS[*]})
  Runtime data:  ${RUNTIME_DIR}
  Accelerator:   ${ACCELERATOR}

Press Ctrl-C to stop the stack. Logs are in ${LOG_DIR}.
EOF

wait "$WEBGUI_PID"
