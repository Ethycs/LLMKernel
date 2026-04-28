"""call_tool input-schema validation per RFC-001 §"Common conventions".

K-MCP V1 mega-round: malformed arguments must surface as JSON-RPC
``-32001`` with a JSON-Pointer-shaped error message.  Validation runs
BEFORE the handler is dispatched, and uses the JSON-Schema in
``_rfc_schemas.TOOL_CATALOG`` (Draft-2020-12).
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict

import pytest
from mcp import types
from mcp.shared.exceptions import McpError

from llm_kernel._rfc_schemas import TOOL_CATALOG, validate_tool_input
from llm_kernel.mcp_server import (
    JSONRPC_SCHEMA_VALIDATION,
    OperatorBridgeServer,
)


ServerHandler = Callable[[types.ClientRequestType], Awaitable[types.ServerResult]]


@pytest.fixture
def bridge() -> OperatorBridgeServer:
    return OperatorBridgeServer(
        agent_id="test-agent",
        zone_id="test-zone",
        trace_id=str(uuid.uuid4()),
    )


def _call_tool_handler(bridge: OperatorBridgeServer) -> ServerHandler:
    handler = bridge.server.request_handlers.get(types.CallToolRequest)
    assert handler is not None
    return handler


async def _invoke(
    bridge: OperatorBridgeServer, name: str, arguments: Dict[str, Any]
) -> types.CallToolResult:
    handler = _call_tool_handler(bridge)
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(request)
    inner = result.root
    assert isinstance(inner, types.CallToolResult)
    return inner


# --- validate_tool_input pure-function unit tests --------------------------


def test_validate_tool_input_accepts_valid_notify() -> None:
    """A well-formed notify call MUST validate."""
    err = validate_tool_input(
        "notify", {"observation": "x", "importance": "info"},
    )
    assert err is None


def test_validate_tool_input_rejects_missing_required() -> None:
    """Missing a required field MUST surface a JSON-Pointer-shaped error."""
    err = validate_tool_input("notify", {"observation": "x"})
    assert err is not None
    assert "importance" in err


def test_validate_tool_input_rejects_wrong_type() -> None:
    """A type mismatch MUST surface a JSON-Pointer-shaped error."""
    err = validate_tool_input(
        "report_progress",
        {"status": "still working", "percent": "not-a-number"},
    )
    assert err is not None
    assert "percent" in err


def test_validate_tool_input_rejects_enum_violation() -> None:
    """An out-of-enum value MUST surface a JSON-Pointer-shaped error."""
    err = validate_tool_input(
        "escalate", {"reason": "boom", "severity": "fatal-emergency"},
    )
    assert err is not None
    assert "severity" in err


def test_validate_tool_input_unknown_tool_returns_none() -> None:
    """An unknown tool MUST return None (routing handles -32601)."""
    err = validate_tool_input("not_a_real_tool", {})
    assert err is None


# --- call_tool integration tests -------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_missing_required_returns_32001(
    bridge: OperatorBridgeServer,
) -> None:
    """call_tool MUST return -32001 on missing required field."""
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "notify", {"observation": "missing-importance"})
    assert exc_info.value.error.code == JSONRPC_SCHEMA_VALIDATION
    assert "importance" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_call_tool_wrong_type_returns_32001(
    bridge: OperatorBridgeServer,
) -> None:
    """A wrong-type field MUST yield -32001."""
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "report_progress",
            {"status": "going", "percent": "fifty"},
        )
    assert exc_info.value.error.code == JSONRPC_SCHEMA_VALIDATION
    assert "percent" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_call_tool_valid_args_dispatches_to_handler(
    bridge: OperatorBridgeServer,
) -> None:
    """Valid args MUST pass validation and dispatch normally."""
    result = await _invoke(
        bridge, "notify", {"observation": "all good", "importance": "info"},
    )
    assert result.structuredContent is not None
    assert result.structuredContent["acknowledged"] is True


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_still_methodnotfound(
    bridge: OperatorBridgeServer,
) -> None:
    """Validation MUST NOT mask -32601 for unknown tools."""
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "fake_tool", {"x": "y"})
    assert exc_info.value.error.code == types.METHOD_NOT_FOUND
