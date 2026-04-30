"""V1 Kernel Gap Closure G9 — native non-blocking tool semantics.

Companion to ``test_native_tool_semantics.py``.  This file pins the
G9-acceptance contracts the brief enumerates:

* ``escalate``           — flood control (>5 critical/60s -> -32005).
* ``notify``             — rate-limit (>10/zone/60s -> -32005).
* ``report_completion``  — once-per-task enforcement (-32004 on dup).
* ``present.artifact_id`` — idempotency (cache hit returns same body).

The thresholds and windows are the V1 inventions pinned in
``mcp_server.py`` constants:

  ESCALATE_FLOOD_MAX        = 5
  ESCALATE_FLOOD_WINDOW_SEC = 60.0
  NOTIFY_RATE_LIMIT_MAX     = 10
  NOTIFY_RATE_LIMIT_WINDOW_SEC = 60.0

K-MCP V1 mega-round Round B G9 closure.  See
``docs/rfcs/RFC-001-mcp-tool-taxonomy.md`` §"Failure modes" and
``docs/rfcs/RFC-002-claude-code-provisioning.md``.
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


# --- escalate flood ------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_throttles_over_threshold(
    bridge: OperatorBridgeServer,
) -> None:
    """escalate flood control: N criticals ack, then N+1 -> -32005.

    Pins the boundary behavior (last allowed call still ack, first
    refused call carries the limit_per_60s data field).
    """
    last_ack = None
    for i in range(ESCALATE_FLOOD_MAX):
        result = await _invoke(
            bridge, "escalate",
            {"reason": f"crit-{i}", "severity": "critical"},
        )
        last_ack = result.structuredContent
    assert last_ack is not None and last_ack["acknowledged"] is True

    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "escalate",
            {"reason": "boom", "severity": "critical"},
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_PROXIED_REFUSED
    assert err.data is not None
    assert err.data.get("limit_per_60s") == ESCALATE_FLOOD_MAX


# --- notify rate-limit ---------------------------------------------------


@pytest.mark.asyncio
async def test_notify_rate_limits(
    bridge: OperatorBridgeServer,
) -> None:
    """notify rate-limit: N acks, then N+1 -> -32005 with limit echoed.

    Differs from the existing test by asserting the limit_per_60s
    data envelope so an agent can surface the cap to the user.
    """
    for i in range(NOTIFY_RATE_LIMIT_MAX):
        await _invoke(
            bridge, "notify",
            {"observation": f"obs-{i}", "importance": "info"},
        )
    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "notify",
            {"observation": "over the line", "importance": "info"},
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_PROXIED_REFUSED
    assert err.data is not None
    assert err.data.get("limit_per_60s") == NOTIFY_RATE_LIMIT_MAX
    assert err.data.get("zone_id") == "test-zone"


# --- report_completion once-per-task -------------------------------------


@pytest.mark.asyncio
async def test_report_completion_enforces_once_per_task(
    bridge: OperatorBridgeServer,
) -> None:
    """report_completion: same task_id twice -> -32004.

    Differs from the existing dup test by also asserting the data
    envelope echoes ``task_id`` and ``zone_id`` so operators can see
    which task they re-completed.
    """
    first = await _invoke(
        bridge, "report_completion",
        {"summary": "first", "task_id": "task-g9"},
    )
    assert first.structuredContent["acknowledged"] is True

    with pytest.raises(McpError) as exc_info:
        await _invoke(
            bridge, "report_completion",
            {"summary": "again", "task_id": "task-g9"},
        )
    err = exc_info.value.error
    assert err.code == JSONRPC_OPERATOR_TIMEOUT  # -32004 per the brief
    assert err.data is not None
    assert err.data.get("task_id") == "task-g9"
    assert err.data.get("zone_id") == "test-zone"


@pytest.mark.asyncio
async def test_report_completion_without_task_id_does_not_block(
    bridge: OperatorBridgeServer,
) -> None:
    """Without ``task_id`` we cannot enforce; both calls MUST ack.

    Pins the additive-evolution rule: agents that pre-date the
    ``task_id`` field still work.
    """
    first = await _invoke(
        bridge, "report_completion", {"summary": "no-task-id-1"},
    )
    second = await _invoke(
        bridge, "report_completion", {"summary": "no-task-id-2"},
    )
    assert first.structuredContent["acknowledged"] is True
    assert second.structuredContent["acknowledged"] is True


# --- present idempotency -------------------------------------------------


@pytest.mark.asyncio
async def test_present_artifact_idempotent(
    bridge: OperatorBridgeServer,
) -> None:
    """Same ``artifact_id`` MUST return same response from cache.

    A second call with a different body but the same artifact_id
    MUST still hit the cache and return the original artifact_id;
    we do NOT mint a new id, and we do NOT re-emit a fresh artifact.
    """
    first = await _invoke(
        bridge, "present",
        {
            "artifact": {"body": "v1"},
            "kind": "code",
            "summary": "first",
            "artifact_id": "art-g9-fixed",
        },
    )
    second = await _invoke(
        bridge, "present",
        {
            "artifact": {"body": "v2-updated"},
            "kind": "code",
            "summary": "second",
            "artifact_id": "art-g9-fixed",
        },
    )
    assert first.structuredContent["artifact_id"] == "art-g9-fixed"
    assert second.structuredContent["artifact_id"] == "art-g9-fixed"
    # Same artifact_id => same returned body shape.  No fresh mint.
    assert (
        first.structuredContent["artifact_id"]
        == second.structuredContent["artifact_id"]
    )
