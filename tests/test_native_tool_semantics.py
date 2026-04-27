"""Native non-blocking tool semantics per RFC-001 §"Failure modes".

K-MCP V1 mega-round: native non-blocking tools enforce four runtime
contracts on top of their previous "always ack" stubs:

* ``escalate`` flood threshold: >5 critical/60s -> -32005.
* ``notify`` rate-limit: >10/zone/60s -> -32005.
* ``report_completion`` once-per-task: second call with same task_id
  -> -32004.
* ``present`` idempotency: same artifact_id returns the same response.

State is kept on the :class:`OperatorBridgeServer` instance under a
non-reentrant lock; tests construct fresh bridges per test for
isolation.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict

import pytest
from mcp import types
from mcp.shared.exceptions import McpError

from llm_kernel.mcp_server import (
    ESCALATE_FLOOD_MAX,
    JSONRPC_OPERATOR_TIMEOUT,
    JSONRPC_PROXIED_REFUSED,
    NOTIFY_RATE_LIMIT_MAX,
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


# --- escalate flood threshold ----------------------------------------------


@pytest.mark.asyncio
async def test_escalate_critical_under_threshold_acknowledges(
    bridge: OperatorBridgeServer,
) -> None:
    """Up to ``ESCALATE_FLOOD_MAX`` critical escalates MUST acknowledge."""
    for i in range(ESCALATE_FLOOD_MAX):
        result = await _invoke(
            bridge, "escalate",
            {"reason": f"crit-{i}", "severity": "critical"},
        )
        assert result.structuredContent["acknowledged"] is True


@pytest.mark.asyncio
async def test_escalate_critical_flood_returns_32005(
    bridge: OperatorBridgeServer,
) -> None:
    """The N+1th critical escalate within the window MUST return -32005."""
    for i in range(ESCALATE_FLOOD_MAX):
        await _invoke(
            bridge, "escalate", {"reason": f"crit-{i}", "severity": "critical"},
        )
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "escalate",
            {"reason": "one too many", "severity": "critical"},
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED
    assert "flood" in exc_info.value.error.message.lower()


@pytest.mark.asyncio
async def test_escalate_non_critical_does_not_count(
    bridge: OperatorBridgeServer,
) -> None:
    """Non-critical escalates MUST NOT count toward the flood threshold."""
    for i in range(ESCALATE_FLOOD_MAX * 3):
        result = await _invoke(
            bridge, "escalate",
            {"reason": f"med-{i}", "severity": "medium"},
        )
        assert result.structuredContent["acknowledged"] is True


# --- notify rate-limit -----------------------------------------------------


@pytest.mark.asyncio
async def test_notify_under_rate_limit_acknowledges(
    bridge: OperatorBridgeServer,
) -> None:
    """Up to ``NOTIFY_RATE_LIMIT_MAX`` notifies MUST acknowledge."""
    for i in range(NOTIFY_RATE_LIMIT_MAX):
        result = await _invoke(
            bridge, "notify",
            {"observation": f"obs-{i}", "importance": "info"},
        )
        assert result.structuredContent["acknowledged"] is True


@pytest.mark.asyncio
async def test_notify_above_rate_limit_returns_32005(
    bridge: OperatorBridgeServer,
) -> None:
    """The N+1th notify in the same zone within the window MUST yield -32005."""
    for i in range(NOTIFY_RATE_LIMIT_MAX):
        await _invoke(
            bridge, "notify", {"observation": f"obs-{i}", "importance": "info"},
        )
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "notify",
            {"observation": "one too many", "importance": "info"},
        )
    assert exc_info.value.error.code == JSONRPC_PROXIED_REFUSED
    assert "rate limit" in exc_info.value.error.message.lower()


# --- report_completion once-per-task ---------------------------------------


@pytest.mark.asyncio
async def test_report_completion_first_call_acknowledges(
    bridge: OperatorBridgeServer,
) -> None:
    """First report_completion with a given task_id MUST acknowledge."""
    result = await _invoke(
        bridge, "report_completion",
        {"summary": "task done", "task_id": "task-1"},
    )
    assert result.structuredContent["acknowledged"] is True


@pytest.mark.asyncio
async def test_report_completion_second_call_returns_32004(
    bridge: OperatorBridgeServer,
) -> None:
    """Second report_completion with same task_id MUST yield -32004."""
    await _invoke(
        bridge, "report_completion",
        {"summary": "first", "task_id": "task-2"},
    )
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "report_completion",
            {"summary": "duplicate", "task_id": "task-2"},
        )
    assert exc_info.value.error.code == JSONRPC_OPERATOR_TIMEOUT
    assert "already" in exc_info.value.error.message.lower()


@pytest.mark.asyncio
async def test_report_completion_distinct_task_ids_each_acknowledge(
    bridge: OperatorBridgeServer,
) -> None:
    """Distinct task_ids MUST each acknowledge."""
    for tid in ("task-A", "task-B", "task-C"):
        result = await _invoke(
            bridge, "report_completion",
            {"summary": tid, "task_id": tid},
        )
        assert result.structuredContent["acknowledged"] is True


# --- present idempotency ---------------------------------------------------


@pytest.mark.asyncio
async def test_present_with_explicit_artifact_id_is_idempotent(
    bridge: OperatorBridgeServer,
) -> None:
    """Repeated present calls with the same artifact_id MUST return the same id."""
    args = {
        "artifact": {"body": "code body"},
        "kind": "code",
        "summary": "demo",
        "artifact_id": "art-fixed-1",
    }
    first = await _invoke(bridge, "present", args)
    second = await _invoke(bridge, "present", args)
    assert first.structuredContent["artifact_id"] == "art-fixed-1"
    assert second.structuredContent["artifact_id"] == "art-fixed-1"


@pytest.mark.asyncio
async def test_present_without_explicit_id_mints_unique(
    bridge: OperatorBridgeServer,
) -> None:
    """When artifact_id is not supplied, each call mints a fresh id."""
    args = {
        "artifact": {"body": "code body"},
        "kind": "code",
        "summary": "demo",
    }
    first = await _invoke(bridge, "present", args)
    second = await _invoke(bridge, "present", args)
    assert first.structuredContent["artifact_id"] != second.structuredContent["artifact_id"]
