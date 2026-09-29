#!/usr/bin/env bash
# Read-only checks for a non-container Oracle Builder demo installation.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ORACLE_DEMO_ENV_FILE:-/etc/oracle-builder/orchestrator.env}"
PROFILE_FILE="${ORACLE_WORKER_DEPLOYMENT_PROFILES:-/etc/oracle-builder/worker-deployments.json}"
DATA_ROOT="${ORACLE_DEMO_DATA_ROOT:-/var/lib/oracle-builder}"
CONTROL_URL="${ORACLE_ORCHESTRATOR_URL:-http://127.0.0.1:8110}"
STRICT=0

if [[ "${1:-}" == "--strict" ]]; then STRICT=1; shift; fi
if [[ "$#" -ne 0 ]]; then echo "usage: $0 [--strict]" >&2; exit 2; fi

failures=0
note() { printf '%s\n' "[check] $*"; }
warn() { printf '%s\n' "[warn] $*" >&2; }
fail() { printf '%s\n' "[fail] $*" >&2; failures=$((failures + 1)); }
require_dir() { [[ -d "$1" ]] && note "directory exists: $1" || fail "missing directory: $1"; }

command -v uv >/dev/null 2>&1 || fail "uv is not on PATH"
command -v curl >/dev/null 2>&1 || fail "curl is not on PATH"
[[ -f "$ENV_FILE" ]] && note "environment file exists: $ENV_FILE" || fail "missing environment file: $ENV_FILE"
[[ -f "$PROFILE_FILE" ]] && note "profile file exists: $PROFILE_FILE" || fail "missing profile file: $PROFILE_FILE"
for name in control artifacts runs datasets worker-identities worker-scratch; do require_dir "$DATA_ROOT/$name"; done

if [[ -f "$PROFILE_FILE" ]] && command -v uv >/dev/null 2>&1; then
  if (cd "$ROOT_DIR" && uv run python -c 'import sys; from oracle_builder.orchestration.cli import load_deployment_profiles; load_deployment_profiles(sys.argv[1])' "$PROFILE_FILE"); then
    note "deployment profile parses"
  else
    fail "deployment profile is invalid"
  fi
fi

if [[ -f "$ENV_FILE" ]]; then
  mode="$(stat -c '%a' "$ENV_FILE" 2>/dev/null || stat -f '%Lp' "$ENV_FILE" 2>/dev/null || true)"
  if [[ "$mode" =~ ^[0-7]{3,4}$ ]] && (( 8#${mode: -2} == 0 )); then note "environment file is not group/world-readable"; else warn "restrict $ENV_FILE to 0600"; fi
fi

if curl --fail --silent --max-time 3 "$CONTROL_URL/health/ready" >/dev/null; then
  note "orchestrator readiness endpoint responds"
elif (( STRICT )); then
  fail "orchestrator is not ready at $CONTROL_URL"
else
  warn "orchestrator is not ready at $CONTROL_URL (expected before first start)"
fi

if (( failures )); then exit 1; fi
note "preflight complete"
