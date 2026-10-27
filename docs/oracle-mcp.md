# Oracle Builder MCP server

Oracle Builder includes an optional Model Context Protocol (MCP) adapter for
AI clients. The adapter translates typed MCP tool calls into requests to the
existing Orchestrator API; it does not replace the Orchestrator or run model
work locally.

The first release uses the official Python SDK v1 (`mcp>=1.28,<2`). The v1
maintenance line supports this repository's Python 3.10 minimum while the
project evaluates the v2 SDK's newer Python requirements. See the [official
Python SDK documentation](https://py.sdk.modelcontextprotocol.io/v1/).

## Start the server

Start an Orchestrator separately, then configure an MCP host with a stdio
server entry. Replace `ABS_REPO` with the absolute path to this checkout:

```json
{
  "mcpServers": {
    "oracle-builder": {
      "command": "uv",
      "args": ["--directory", "ABS_REPO", "run", "--extra", "mcp", "oracle-mcp"],
      "env": {
        "ORACLE_ORCHESTRATOR_URL": "http://127.0.0.1:8110"
      }
    }
  }
}
```

The server defaults to `http://127.0.0.1:8110` and reads
`ORACLE_ORCHESTRATOR_URL` when supplied. Use the `--timeout` flag to override
the 30-second request timeout. The adapter uses stdio only in this release.

Install the optional dependency and inspect the CLI with:

```bash
uv sync --extra mcp
uv run --extra mcp oracle-mcp --help
```

Reads are available by default. Writes are deliberately opt-in for each
server process and still pass through the Orchestrator's role authorization:

```json
{
  "mcpServers": {
    "oracle-builder-writes": {
      "command": "uv",
      "args": ["--directory", "ABS_REPO", "run", "--extra", "mcp", "oracle-mcp", "--allow-writes"],
      "env": {
        "ORACLE_ORCHESTRATOR_URL": "http://127.0.0.1:8110",
        "ORACLE_ORCHESTRATOR_TOKEN": "YOUR_OPERATOR_TOKEN"
      }
    }
  }
}
```

Do not put a real token in a checked-in configuration file. The token is sent
as a Bearer credential to the API, and the API remains the authority for
operator/admin permissions.

## Tools, resources, and prompts

The read-only surface includes exactly four tools: `health`, `list_records`,
`get_record`, and `get_config_schema`. `list_records` accepts a bounded
collection enum for datasets, model definitions, model templates, queued runs,
jobs, inference runs, post-hoc runs, artifacts, worker pools, workers,
operations, specifications, experiments, recipes, and training catalog entries.
It returns a page (default 20, maximum 100) from a live listing. Inference,
post-hoc, and operation listings are limited upstream to the newest 500 records
and report `source_may_be_truncated` when that window is full. `get_record`
supports the server's typed kinds and optional views using record IDs. API
records can contain server-side path metadata; tools do not accept local file
paths or read those files.

With `--allow-writes` and a token, the server registers 14 additional typed
operations (18 tools total):
`create_model_definition`, `update_model_definition`, `queue_training`,
`verify_training`, `start_training`, `cancel_queued_run`, `create_inference`,
`start_inference`, `cancel_inference`, `create_posthoc`, `start_posthoc`,
`cancel_posthoc`, `control_job`, and `cancel_job` (with separate typed
operations for each lifecycle action). Queue verification and start remain
separate calls so an agent cannot accidentally launch unverified work in a
single opaque operation.

The `oracle://guide` and `oracle://capabilities` resources provide stable
workflow and capability context. The `plan_training` prompt helps an agent
produce a reviewable training plan before it invokes write tools.

The adapter intentionally has no direct filesystem, upload/download, worker
credential, shell, or arbitrary endpoint tool. Use the Orchestrator's existing
API and CLI for those boundaries.

## Implementation plan

The implemented first phase is a thin, typed stdio bridge with read-only
defaults, explicit write registration, bounded collection and operation enums,
and tests for request routing, pagination, error handling, and write gating.
Acceptance is: `oracle-mcp --help` works with the extra installed; default
tool discovery exposes four tools; write discovery requires both
`--allow-writes` and a token; collection pagination is bounded; and each write
maps to the existing Orchestrator endpoint without local execution.

A concrete training workflow is: (1) read health and discover records, (2)
queue training, (3) verify the queued run, (4) inspect the returned operation
and queued-run state, (5) start only after verification succeeds, then poll
the durable run/job records. A second phase can add authenticated Streamable
HTTP after deployment and identity policy are selected. A later phase can add
richer resources (event streams and artifact evidence) without widening the
mutation boundary.

The existing Orchestrator service must be running before an MCP client calls a
tool. Starting `oracle-mcp` launches only the adapter process.

Run the adapter's protocol, routing, and real-API integration tests with:

```bash
uv run --extra mcp pytest tests/test_mcp_server.py tests/test_mcp_integration.py
```
