#!/usr/bin/env sh
# Prepare the host-owned state tree and an exact-path Docker worker profile.
# Run once before `docker compose --profile image build`; it never handles raw
# tokens and refuses to overwrite an existing generated profile.
set -eu

root=${1:-}
if [ -z "$root" ] || [ "${root#/*}" = "$root" ]; then
  echo "usage: $0 ABSOLUTE_DATA_ROOT" >&2
  exit 2
fi
case "$root" in
  *[!A-Za-z0-9._/-]*) echo "data root may contain only letters, numbers, ., _, /, and -" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$root")" && pwd -P)/$(basename "$root")
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
runtime="$here/runtime"
profile="$runtime/worker-deployments.json"

if [ -e "$profile" ]; then
  echo "refusing to overwrite $profile; remove it only when intentionally changing the host root" >&2
  exit 1
fi

umask 077
mkdir -p "$root/control" "$root/artifacts" "$root/runs" "$root/datasets" \
  "$root/workspace" "$root/logs" "$root/worker-identities" \
  "$root/worker-scratch" "$root/worker-bootstrap" "$runtime"

sed "s|REPLACE_WITH_ABSOLUTE_HOST_DATA_ROOT|$root|g" \
  "$here/worker-deployments.docker.json.example" > "$profile"
chmod 600 "$profile"

echo "Prepared $root"
echo "Generated $profile"
echo "Next: copy .env.example to .env, configure tokens, then run docker compose --profile image build."
