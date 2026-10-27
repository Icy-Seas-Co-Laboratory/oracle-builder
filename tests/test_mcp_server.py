import asyncio
import json
import os
import sys

import httpx
import pytest

mcp = pytest.importorskip("mcp")

from mcp.server.fastmcp.exceptions import ToolError

from oracle_builder.mcp.server import create_server


def _run(awaitable):
    return asyncio.run(awaitable)


def _text(result):
    if isinstance(result, tuple):
        result = result[0]
    content = getattr(result, "content", result)
    for item in content if isinstance(content, list) else []:
        text = getattr(item, "text", None)
        if text is not None:
            return text
    return ""


def _data(result):
    text = _text(result)
    return json.loads(text) if text else result


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_read_only_surface_tools_resources_and_prompt():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/jobs":
            return httpx.Response(200, json={"jobs": [{"job_id": "a"}, {"job_id": "b"}]})
        return httpx.Response(200, json={})

    server = create_server("http://orch.test", client=_client(handler))
    tools = _run(server.list_tools())
    names = {tool.name for tool in tools}
    assert names == {"health", "list_records", "get_record", "get_config_schema"}
    health = _data(_run(server.call_tool("health", {})))
    assert health == {"status": "ready"}
    page = _data(_run(server.call_tool("list_records", {"collection": "jobs", "limit": 1})))
    assert page["items"] == [{"job_id": "a"}] and page["has_more"]
    assert calls[0].headers.get("authorization") is None

    resources = _run(server.list_resources())
    assert {str(resource.uri) for resource in resources} == {"oracle://guide", "oracle://capabilities"}
    guide = _run(server.read_resource("oracle://guide"))
    assert "verify_training" in str(guide)
    prompts = _run(server.list_prompts())
    assert {prompt.name for prompt in prompts} == {"plan_training"}
    prompt = _run(server.get_prompt("plan_training", {"goal": "compare models"}))
    assert "compare models" in str(prompt)


def test_read_only_validation_rejects_invalid_views_and_ids_without_http():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"jobs": []})

    server = create_server(client=_client(handler))
    with pytest.raises(ToolError):
        _run(server.call_tool("get_record", {"kind": "job", "id": "bad id", "view": "summary"}))
    with pytest.raises(ToolError):
        _run(server.call_tool("get_record", {"kind": "dataset", "id": "x", "view": "events"}))
    assert calls == []


def test_write_tools_require_opt_in_and_token():
    with pytest.raises(ValueError, match="allow-writes"):
        create_server(allow_writes=True)
    server = create_server(client=_client(lambda request: httpx.Response(200, json={})))
    # Mutation tools are absent in the default read-only profile.
    assert "queue_training" not in {tool.name for tool in _run(server.list_tools())}


def test_write_routing_validation_headers_and_explicit_training_flow():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/v1/model-definitions/x:queue":
            return httpx.Response(201, json={"queued_run_id": "q1"})
        if request.url.path == "/v1/queued-runs:verify":
            return httpx.Response(202, json={"operation": {"operation_id": "op1"}})
        return httpx.Response(200, json={"ok": True})

    server = create_server("http://orch.test", token="secret", allow_writes=True, client=_client(handler))
    names = {tool.name for tool in _run(server.list_tools())}
    assert {"queue_training", "verify_training", "start_training", "create_inference", "control_job"} <= names
    queued = _data(_run(server.call_tool("queue_training", {
        "definition_id": "x", "name": "run", "dataset_id": "d", "worker_pool_id": "p",
        "auto_batch_size": True,
    })))
    assert queued["queued_run_id"] == "q1"
    verified = _data(_run(server.call_tool("verify_training", {"queued_run_ids": ["q1"]})))
    assert verified["operation"]["operation_id"] == "op1"
    assert json.loads(seen[0].content)["batch_size_mode"] == "auto"
    assert json.loads(seen[1].content)["start_after_verification"] is False
    assert all(request.headers["authorization"] == "Bearer secret" for request in seen)
    with pytest.raises(ToolError):
        _run(server.call_tool("queue_training", {
            "definition_id": "x", "name": "run", "dataset_id": "d", "worker_pool_id": "p",
            "batch_size": 4, "auto_batch_size": True,
        }))
    with pytest.raises(ToolError):
        _run(server.call_tool("verify_training", {"queued_run_ids": []}))


def test_write_errors_are_safe_and_redirects_are_not_followed():
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        if request.url.path == "/v1/jobs/j:cancel":
            if calls > 1:
                return httpx.Response(302, headers={"location": "https://evil.test"})
            return httpx.Response(500, text="secret")

    server = create_server("http://orch.test", token="secret", allow_writes=True, client=_client(handler))
    with pytest.raises(ToolError, match="HTTP 500") as exc:
        _run(server.call_tool("cancel_job", {"id": "j"}))
    assert "secret" not in str(exc.value)
    with pytest.raises(ToolError, match="HTTP 302"):
        _run(server.call_tool("cancel_job", {"id": "j"}))


@pytest.mark.parametrize("token", ["operator-secret", 'secret-"quoted"'])
def test_json_error_redacts_token(token):
    with _client(lambda request: httpx.Response(401, json={"detail": f"Rejected {token}"})) as client:
        server = create_server(token=token, client=client)
        with pytest.raises(ToolError, match="HTTP 401") as error:
            _run(server.call_tool("health", {}))
        assert token not in str(error.value)
        assert "[redacted]" in str(error.value)


def test_ambiguous_write_timeout_is_not_retried():
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("sensitive transport diagnostics", request=request)

    with _client(handler) as client:
        server = create_server(token="secret", allow_writes=True, client=client)
        with pytest.raises(ToolError, match="outcome may be unknown") as error:
            _run(server.call_tool("start_inference", {"id": "run"}))
        assert "sensitive" not in str(error.value)
        assert len(seen) == 1


def test_stdio_protocol_handshake_and_read_only_boundary():
    """Exercise the actual MCP wire protocol, without starting an Orchestrator."""
    anyio = pytest.importorskip("anyio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        env = dict(os.environ)
        env.pop("ORACLE_ORCHESTRATOR_TOKEN", None)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "oracle_builder.mcp.server", "--url", "http://127.0.0.1:1", "--timeout", "0.2"],
            env=env,
        )
        with anyio.fail_after(10):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert {tool.name for tool in tools.tools} == {"health", "list_records", "get_record", "get_config_schema"}
                    resources = await session.list_resources()
                    assert {str(item.uri) for item in resources.resources} == {"oracle://guide", "oracle://capabilities"}
                    guide = await session.read_resource("oracle://guide")
                    assert "verify_training" in str(guide)
                    prompts = await session.list_prompts()
                    assert [item.name for item in prompts.prompts] == ["plan_training"]
                    unknown = await session.call_tool("queue_training", {"definition_id": "x"})
                    assert unknown.isError

    _run(scenario())


@pytest.mark.parametrize(
    ("url", "timeout", "allow_writes", "token", "match"),
    [
        ("ftp://orch.test", 1, False, None, r"HTTP\(S\)"),
        ("http://user:pass@orch.test", 1, False, None, "credentials"),
        ("http://orch.test/path?x=1", 1, False, None, "query"),
        ("http://orch.test", 0, False, None, "finite"),
        ("http://orch.test", float("nan"), False, None, "finite"),
        ("http://orch.test", 1, True, " ", "allow-writes"),
    ],
)
def test_mcp_settings_validation(url, timeout, allow_writes, token, match):
    with pytest.raises(ValueError, match=match):
        create_server(url, timeout=timeout, allow_writes=allow_writes, token=token)


def test_all_collection_listings_use_expected_api_keys():
    expected = {
        "datasets": "datasets", "model_definitions": "definitions", "model_templates": "templates",
        "queued_runs": "queued_runs", "jobs": "jobs", "inference_runs": "inference_runs",
        "posthoc_runs": "posthoc_runs", "artifacts": "artifacts", "worker_pools": "pools",
        "workers": "workers", "operations": "operations", "specifications": "specifications",
        "experiments": "experiments", "recipes": "recipes", "training_catalog": "entries",
    }
    for collection, key in expected.items():
        def handler(request, key=key):
            return httpx.Response(200, json={key: [{"id": "a"}]})
        server = create_server(client=_client(handler))
        result = _data(_run(server.call_tool("list_records", {"collection": collection})))
        assert result["items"] == [{"id": "a"}]


@pytest.mark.parametrize(
    ("tool", "arguments", "method", "path", "body_keys"),
    [
        ("create_model_definition", {"name": "m", "template_id": "t", "config": {}}, "POST", "/v1/model-definitions", {"name", "template_id", "config", "description"}),
        ("update_model_definition", {"id": "m", "expected_revision": 1, "config": {}}, "PATCH", "/v1/model-definitions/m", {"expected_revision", "config"}),
        ("queue_training", {"definition_id": "m", "name": "r", "dataset_id": "d", "worker_pool_id": "p"}, "POST", "/v1/model-definitions/m:queue", {"name", "dataset_id", "worker_pool_id", "batch_size_mode"}),
        ("verify_training", {"queued_run_ids": ["q"]}, "POST", "/v1/queued-runs:verify", {"queued_run_ids", "start_after_verification"}),
        ("start_training", {"queued_run_ids": ["q"], "worker_pool_id": "p"}, "POST", "/v1/queued-runs:start", {"queued_run_ids", "worker_pool_id"}),
        ("cancel_queued_run", {"id": "q"}, "POST", "/v1/queued-runs/q:cancel", set()),
        ("create_inference", {"name": "i", "model_artifact_id": "a", "dataset_id": "d", "worker_pool_id": "p"}, "POST", "/v1/inference-runs", {"name", "model_artifact_id", "dataset_id", "worker_pool_id"}),
        ("start_inference", {"id": "i"}, "POST", "/v1/inference-runs/i:start", set()),
        ("cancel_inference", {"id": "i"}, "POST", "/v1/inference-runs/i:cancel", set()),
        ("create_posthoc", {"name": "p", "model_artifact_id": "a", "dataset_id": "d", "worker_pool_id": "w"}, "POST", "/v1/posthoc-runs", {"name", "model_artifact_id", "dataset_id", "worker_pool_id"}),
        ("start_posthoc", {"id": "p"}, "POST", "/v1/posthoc-runs/p:start", set()),
        ("cancel_posthoc", {"id": "p"}, "POST", "/v1/posthoc-runs/p:cancel", set()),
        ("control_job", {"job_id": "j", "action": "pause", "command_id": "c"}, "POST", "/v1/jobs/j/commands", {"action", "command_id"}),
        ("cancel_job", {"id": "j"}, "POST", "/v1/jobs/j:cancel", set()),
    ],
)
def test_every_mutation_tool_routes_once(tool, arguments, method, path, body_keys):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})
    server = create_server("http://orch.test", token="tok", allow_writes=True, client=_client(handler))
    _run(server.call_tool(tool, arguments))
    assert len(seen) == 1
    assert seen[0].method == method and seen[0].url.path == path
    if body_keys:
        assert body_keys <= set(json.loads(seen[0].content))
