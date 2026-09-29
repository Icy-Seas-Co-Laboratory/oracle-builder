# syntax=docker/dockerfile:1
# One reproducible Python image supplies the Orchestrator and the fixed
# pull-worker executable. The Orchestrator target also includes the Docker
# client because it may manage approved Docker worker profiles through the
# host's Docker socket.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/oracle-builder/.venv \
    PATH=/opt/oracle-builder/.venv/bin:$PATH

WORKDIR /opt/oracle-builder

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /usr/local/bin/
COPY . .

# The production image includes the S3 client so object replication can be
# enabled only through environment/configuration; it does not require S3.
RUN uv sync --frozen --no-dev --extra storage-s3

FROM base AS worker
CMD ["oracle-worker"]

FROM base AS orchestrator
# The Docker provider calls a fixed argument vector through the host socket.
# No Docker daemon runs in this container.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --chmod=755 deploy/docker/orchestrator-entrypoint.sh /usr/local/bin/oracle-orchestrator-container
ENTRYPOINT ["/usr/local/bin/oracle-orchestrator-container"]
