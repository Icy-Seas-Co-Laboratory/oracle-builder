#!/bin/sh
set -eu

data_root="${ORACLE_DATA_ROOT:-/var/lib/oracle-builder}"
args="--database ${data_root}/control/orchestrator.sqlite --workspace-root ${data_root}/workspace --artifact-root ${data_root}/artifacts --runs-root ${data_root}/runs --datasets-root ${data_root}/datasets --log-root ${data_root}/logs --host 0.0.0.0 --port 8110 --upload-limit-mib ${ORACLE_UPLOAD_LIMIT_MIB:-10240}"

if [ -n "${ORACLE_WORKER_DEPLOYMENT_PROFILES:-}" ]; then
  args="${args} --worker-deployment-profiles ${ORACLE_WORKER_DEPLOYMENT_PROFILES}"
fi
if [ -n "${ORACLE_ARTIFACT_S3_BUCKET:-}" ]; then
  args="${args} --s3-bucket ${ORACLE_ARTIFACT_S3_BUCKET} --s3-prefix ${ORACLE_ARTIFACT_S3_PREFIX:-oracle-builder}"
  if [ -n "${ORACLE_ARTIFACT_S3_ENDPOINT_URL:-}" ]; then
    args="${args} --s3-endpoint-url ${ORACLE_ARTIFACT_S3_ENDPOINT_URL}"
  fi
  if [ -n "${ORACLE_ARTIFACT_S3_REGION:-}" ]; then
    args="${args} --s3-region ${ORACLE_ARTIFACT_S3_REGION}"
  fi
fi

# Values above are operator-controlled environment configuration, never API
# input. They contain no raw role tokens; the CLI reads only their SHA-256 map.
# shellcheck disable=SC2086
exec oracle-orchestrator ${args}
