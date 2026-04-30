"""V1 Kernel Gap Closure — MCP tool input schema validation (-32001).

Companion to ``test_tool_schema_validation.py``.  This file pins the
brief-named acceptance scenarios for ``validate_tool_input`` and the
``call_tool`` integration:

* valid args -> validator returns ``None``
* invalid args -> validator returns a non-None error string
* call_tool with invalid args -> raises ``McpError`` with code
  ``-32001`` (RFC-001 §"Common conventions").

K-MCP V1 mega-round Round B closure.  See
``docs/atoms/protocols/mcp-tool-call.md`` and
``docs/rfcs/RFC-001-mcp-tool-taxonomy.md``.
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


ServerHandler = Callable[[types.ClientRequest], Awaitable[types.ServerResult]]


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
    server_result = await handler(request)
    inner = server_result.root
    assert isinstance(inner, types.CallToolResult)
    return inner


# --- Pure-function validator contract ------------------------------------


def test_valid_tool_args_return_none_from_validator() -> None:
    """validate_tool_input MUST return None on schema-valid args.

    Sweeps every tool in the V1 catalog using a minimum-required
    payload synthesized from the schema's ``required`` list, picking
    a representative value per JSON Schema ``type``.
    """
    # Minimum-valid payloads per tool (covers all required fields).
    fixtures: Dict[str, Dict[str, Any]] = {
        "ask": {"question": "q?"},
        "clarify": {
            "question": "pick one",
            "options": [
                {"id": "a_1", "label": "A"},
                {"id": "b_2", "label": "B"},
            ],
        },
        "propose": {"action": "do", "rationale": "because"},
        "request_approval": {
            "action": "do",
            "diff_preview": {"kind": "text", "body": "x"},
            "risk_level": "low",
        },
        "report_progress": {"status": "working"},
        "report_completion": {"summary": "done"},
        "report_problem": {"severity": "info", "description": "fyi"},
        "present": {
            "artifact": {"body": "x"},
            "kind": "code",
            "summary": "demo",
        },
        "notify": {"observation": "x", "importance": "info"},
        "escalate": {"reason": "x", "severity": "high"},
        "read_file": {"path": "x.txt"},
        "write_file": {"path": "x.txt", "content": "y"},
        "run_command": {"command": "echo"},
    }
    for tool_name in TOOL_CATALOG:
        assert tool_name in fixtures, (
            f"missing minimum-valid fixture for {tool_name!r}"
        )
        err = validate_tool_input(tool_name, fixtures[tool_name])
        assert err is None, (
            f"valid args for {tool_name!r} should pass; got error: {err!r}"
        )


def test_invalid_tool_args_return_error_string() -> None:
    """validate_tool_input MUST return an error string on violation.

    Covers the four common violation classes:

    1. missing required field
    2. wrong type
    3. enum violation
    4. minLength violation
    """
    # 1. missing required
    err = validate_tool_input("propose", {"action": "x"})  # missing rationale
    assert isinstance(err, str)
    assert "rationale" in err

    # 2. wrong type
    err = validate_tool_input(
        "report_progress",
        {"status": "still working", "percent": "halfway"},
    )
    assert isinstance(err, str)
    assert "percent" in err

    # 3. enum violation
    err = validate_tool_input(
        "report_problem",
        {"severity": "critical", "description": "bug"},  # only fatal/error/warning/info
    )
    assert isinstance(err, str)
    assert "severity" in err

    # 4. minLength
    err = validate_tool_input(
        "ask", {"question": ""},  # minLength=1
    )
    assert isinstance(err, str)
    assert "question" in err


# --- call_tool integration: -32001 envelope ------------------------------


@pytest.mark.asyncio
async def test_call_tool_with_invalid_args_emits_minus_32001(
    bridge: OperatorBridgeServer,
) -> None:
    """call_tool MUST raise McpError(-32001) on schema violation.

    Asserts:
    * the error code is exactly ``-32001`` (JSONRPC_SCHEMA_VALIDATION)
    * the data envelope includes ``tool_name`` + ``agent_id`` so the
      operator surface can attribute the violation to its source
    * the message body mentions the offending field path
    """
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "escalate",
            {"reason": "x", "severity": "extra-spicy"},  # not in enum
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_SCHEMA_VALIDATION
    assert "severity" in err.message
    assert err.data is not None
    assert err.data.get("tool_name") == "escalate"
    assert err.data.get("agent_id") == "test-agent"


@pytest.mark.asyncio
async def test_call_tool_validation_runs_before_handler(
    bridge: OperatorBridgeServer,
) -> None:
    """Validation MUST short-circuit before the handler runs.

    A read_file with a missing required ``path`` MUST yield -32001
    rather than reach the proxied I/O path (which would yield
    -32005 or -32006).  This pins ordering: schema check first.
    """
    with pytest.raises(McpError) as exc_info:
        await _invoke(bridge, "read_file", {})  # missing path
    assert exc_info.value.error.code == JSONRPC_SCHEMA_VALIDATION
    assert "path" in exc_info.value.error.message
