"""Exercise MCP tools against the real API, auth middleware and isolated SQLite state."""
import asyncio
from hashlib import sha256
from pathlib import Path
import shutil

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

from oracle_builder.mcp.server import COLLECTION_PATHS, create_server
from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    # SDK structured-output tools return (content, structured_content).
    return result[1]


def test_mcp_discovers_real_api_and_preserves_definition_revisions(tmp_path):
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", tmp_path / "configs")
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=tmp_path)
    token = "test-operator-token"
    app = create_app(orchestrator, role_tokens={"operator": sha256(token.encode()).hexdigest()})
    with TestClient(app) as client:
        server = create_server("http://testserver", client=client, token=token, allow_writes=True)
        for collection in COLLECTION_PATHS:
            page = call(server, "list_records", {"collection": collection})
            assert isinstance(page["items"], list), collection
        templates = call(server, "list_records", {"collection": "model_templates"})
        assert templates["items"]
        definition = call(server, "create_model_definition", {
            "name": "MCP baseline", "template_id": templates["items"][0]["template_id"], "config": {},
        })
        updated = call(server, "update_model_definition", {
            "id": definition["definition_id"], "expected_revision": 1,
            "name": "MCP baseline revised", "config": definition["config"],
        })
        assert updated["revision"] == 2
        with pytest.raises(ToolError, match="HTTP 409"):
            call(server, "update_model_definition", {
                "id": definition["definition_id"], "expected_revision": 1, "config": definition["config"],
            })
        detail = call(server, "get_record", {"kind": "model_definition", "id": definition["definition_id"]})
        assert detail["name"] == "MCP baseline revised"
        revisions = call(server, "get_record", {
            "kind": "model_definition", "id": definition["definition_id"], "view": "revisions",
        })
        assert len(revisions["revisions"]) == 2


def test_mcp_write_opt_in_does_not_bypass_orchestrator_roles(tmp_path):
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=tmp_path)
    analyst = "test-analyst-token"
    app = create_app(orchestrator, role_tokens={"analyst": sha256(analyst.encode()).hexdigest()})
    with TestClient(app) as client:
        server = create_server("http://testserver", client=client, token=analyst, allow_writes=True)
        with pytest.raises(ToolError, match="HTTP 401"):
            call(server, "create_model_definition", {"name": "unauthorized", "template_id": "resnet18", "config": {}})
    assert orchestrator.model_definitions() == []
