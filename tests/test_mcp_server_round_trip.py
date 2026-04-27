"""Track B1 contract test for RFC-001 v1.0.0 MCP tool catalog.

This test exercises the in-process tool registry of
:class:`llm_kernel.mcp_server.OperatorBridgeServer` without going over
stdio. It is the deliverable acceptance test for Stage 2 Track B1: the
server starts, registers all thirteen RFC-001 tools, and round-trips a
``notify`` plus ``report_completion`` end-to-end.

Run with::

    pixi run -e kernel pytest vendor/LLMKernel/tests/test_mcp_server_round_trip.py -v
"""

from __future__ import annotations

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
    """notify MUST acknowledge with a UUID run_id and the RFC version."""
    result = await _invoke(
        bridge, "notify", {"observation": "hello", "importance": "info"}
    )
    payload = _structured(result)
    assert payload["acknowledged"] is True
    # run_id MUST be a UUID
    uuid.UUID(payload["run_id"])
    assert payload["_rfc_version"] == RFC_001_VERSION


@pytest.mark.asyncio
async def test_report_completion_round_trip(
    bridge: OperatorBridgeServer,
) -> None:
    """report_completion MUST return a structured success envelope."""
    result = await _invoke(bridge, "report_completion", {"summary": "test done"})
    payload = _structured(result)
    assert payload["acknowledged"] is True
    uuid.UUID(payload["run_id"])
    assert payload["_rfc_version"] == RFC_001_VERSION


@pytest.mark.asyncio
async def test_read_file_is_unimplemented(bridge: OperatorBridgeServer) -> None:
    """Proxied read_file MUST raise NotImplementedError until B2/B5 land."""
    with pytest.raises(NotImplementedError, match="Track B1 stub"):
        await _invoke(bridge, "read_file", {"path": "anything.txt"})


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
