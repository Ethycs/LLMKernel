"""Track B1 contract test for RFC-001 v1.0.0 MCP tool catalog.

This test exercises the in-process tool registry of
:class:`llm_kernel.mcp_server.OperatorBridgeServer` without going over
stdio. It is the deliverable acceptance test for Stage 2 Track B1: the
server starts, registers all thirteen RFC-001 tools, and round-trips a
``notify`` plus ``report_completion`` end-to-end.

Post-OTLP refactor (R1-K) the bridge surfaces the OTLP ``spanId``
(16 lowercase hex chars) as the structured-content ``run_id`` so the
operator surface still has a stable identifier per call.

Run with::

    pixi run -e kernel pytest vendor/LLMKernel/tests/test_mcp_server_round_trip.py -v
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Awaitable, Callable, Dict

import pytest
from mcp import types
from mcp.shared.exceptions import McpError

from llm_kernel.mcp_server import (
    RFC_001_VERSION,
    SERVER_NAME,
    OperatorBridgeServer,
)

#: 16-lowercase-hex regex (OTLP spanId).
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")

ServerHandler = Callable[[types.ClientRequestType], Awaitable[types.ServerResult]]


@pytest.fixture
def bridge() -> OperatorBridgeServer:
    """Build a fresh OperatorBridgeServer with deterministic ids."""
    return OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )


def _call_tool_handler(bridge: OperatorBridgeServer) -> ServerHandler:
    """Return the registered tools/call handler from the wrapped MCP server."""
    handler = bridge.server.request_handlers.get(types.CallToolRequest)
    assert handler is not None, "call_tool handler should be registered"
    return handler


async def _invoke(
    bridge: OperatorBridgeServer, name: str, arguments: Dict[str, Any]
) -> types.CallToolResult:
    """Invoke a tool through the MCP request handler and return its CallToolResult."""
    handler = _call_tool_handler(bridge)
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    server_result = await handler(request)
    inner = server_result.root
    assert isinstance(inner, types.CallToolResult), (
        f"expected CallToolResult, got {type(inner).__name__}"
    )
    return inner


def _structured(result: types.CallToolResult) -> Dict[str, Any]:
    """Return the structured payload of a CallToolResult."""
    assert result.structuredContent is not None, "tool MUST return structured content"
    return result.structuredContent


def test_server_identity(bridge: OperatorBridgeServer) -> None:
    """The MCP server name MUST equal RFC-002's stable identifier."""
    assert bridge.server.name == SERVER_NAME == "llmkernel-operator-bridge"


@pytest.mark.asyncio
async def test_catalog_lists_thirteen_tools(bridge: OperatorBridgeServer) -> None:
    """All thirteen RFC-001 tools MUST be registered."""
    handler = bridge.server.request_handlers[types.ListToolsRequest]
    request = types.ListToolsRequest(method="tools/list")
    server_result = await handler(request)
    tools = server_result.root.tools
    assert len(tools) == 13
    names = {tool.name for tool in tools}
    expected = {
        "ask", "clarify", "propose", "request_approval",
        "report_progress", "report_completion", "report_problem",
        "present", "notify", "escalate",
        "read_file", "write_file", "run_command",
    }
    assert names == expected


@pytest.mark.asyncio
async def test_notify_round_trip(bridge: OperatorBridgeServer) -> None:
    """notify MUST acknowledge with an OTLP spanId run_id and the RFC version."""
    result = await _invoke(
        bridge, "notify", {"observation": "hello", "importance": "info"}
    )
    payload = _structured(result)
    assert payload["acknowledged"] is True
    # run_id MUST be a 16-lowercase-hex OTLP spanId.
    assert _SPAN_ID_RE.match(payload["run_id"]), (
        f"expected 16-hex spanId, got {payload['run_id']!r}"
    )
    assert payload["_rfc_version"] == RFC_001_VERSION


@pytest.mark.asyncio
async def test_report_completion_round_trip(
    bridge: OperatorBridgeServer,
) -> None:
    """report_completion MUST return a structured success envelope."""
    result = await _invoke(bridge, "report_completion", {"summary": "test done"})
    payload = _structured(result)
    assert payload["acknowledged"] is True
    assert _SPAN_ID_RE.match(payload["run_id"]), (
        f"expected 16-hex spanId, got {payload['run_id']!r}"
    )
    assert payload["_rfc_version"] == RFC_001_VERSION


@pytest.mark.asyncio
async def test_read_file_round_trip(
    bridge: OperatorBridgeServer, tmp_path, monkeypatch,
) -> None:
    """Proxied read_file returns workspace-rooted file contents.

    Per RFC-001 §read_file the proxied handler returns the file's
    contents as a string with the encoding, byte size, and a
    truncation flag.  The B1 ``NotImplementedError`` stub was
    replaced by K-MCP under the V1 mega-round.
    """
    monkeypatch.setenv("LLMKERNEL_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "hello.txt"
    target.write_text("hello round-trip", encoding="utf-8")
    result = await _invoke(bridge, "read_file", {"path": "hello.txt"})
    payload = _structured(result)
    assert payload["content"] == "hello round-trip"
    assert payload["encoding"] == "utf-8"
    assert payload["truncated"] is False
    assert payload["size_bytes"] == len("hello round-trip")


@pytest.mark.asyncio
async def test_unknown_tool_returns_method_not_found(
    bridge: OperatorBridgeServer,
) -> None:
    """An unknown tool MUST surface JSON-RPC -32601 method-not-found.

    Per RFC-001 §Failure modes the kernel rejects unknown tools without
    invoking any handler; the McpError raised here carries
    ``code = types.METHOD_NOT_FOUND``.
    """
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "fake_tool", {})
    assert exc_info.value.error.code == types.METHOD_NOT_FOUND == -32601


@pytest.mark.asyncio
async def test_unknown_tool_emits_agent_emit_invalid_tool_use() -> None:
    """RFC-002 §"Failure modes" / RFC-005 §"`agent_emit` runs": the kernel
    additionally surfaces an ``agent_emit:invalid_tool_use`` span so the
    operator surface sees the bypass attempt rather than just the silent
    JSON-RPC denial.
    """
    from typing import Any, Dict, List
    from llm_kernel.run_tracker import RunTracker
    from llm_kernel._attrs import decode_attrs

    class _ListSink:
        def __init__(self) -> None:
            self.envelopes: List[Dict[str, Any]] = []
        def emit(self, env: Dict[str, Any]) -> None:
            self.envelopes.append(env)

    sink = _ListSink()
    tracker = RunTracker(
        trace_id=str(uuid.uuid4()), sink=sink,
        agent_id="test-agent", zone_id="test-zone",
    )
    bridge = OperatorBridgeServer(
        agent_id="test-agent", zone_id="test-zone",
        trace_id=str(uuid.uuid4()), run_tracker=tracker,
    )
    with pytest.raises(McpError):
        await _invoke(bridge, "fake_tool", {"x": 1})
    # The agent_emit:invalid_tool_use span MUST exist with the
    # documented attributes.
    invalid_spans = [
        s for s in tracker.iter_runs() if s.name == "agent_emit:invalid_tool_use"
    ]
    assert len(invalid_spans) == 1
    attrs = decode_attrs(invalid_spans[0].attributes)
    assert attrs["llmnb.run_type"] == "agent_emit"
    assert attrs["llmnb.emit_kind"] == "invalid_tool_use"
    assert "fake_tool" in attrs["llmnb.emit_content"]
    assert "fake_tool" in attrs["llmnb.parser_diagnostic"]
